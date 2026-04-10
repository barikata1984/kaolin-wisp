# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modifications Copyright (c) 2026, NeuS2 trainer for kaolin-wisp.

import os
import logging as log
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Optional

import wisp
from wisp.config import configure, autoconfig, instantiate
from wisp.trainers import BaseTrainer, ConfigBaseTrainer
from wisp.trainers.tracker import Tracker
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.datasets import MultiviewDataset
from wisp.datasets.transforms import SampleRays
from wisp.core import Rays, RenderBuffer
from wisp.ops.differential.gradients import finitediff_gradient

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@configure
class ConfigNeuSTrainer(ConfigBaseTrainer):
    prune_every: int = 100
    """ Invokes nef.prune() logic every "prune_every" iterations. """

    random_lod: bool = False
    """ If True, random LODs will be picked per step. """

    rgb_lambda: float = 1.0
    """ Loss weight for RGB loss. """

    rgb_loss_type: str = 'l1'
    """ RGB loss type: 'l1', 'l2', 'huber'. """

    rgb_loss_denom: str = 'rays'
    """ Denominator for RGB loss normalization: 'rays' or 'samples'. """

    eikonal_lambda: float = 0.1
    """ Loss weight for Eikonal regularization. """

    progressive_lod_every: int = -1
    """ Add one LOD level every N iterations. -1 to auto-compute from max_epochs. """

    target_sample_size: int = 2 ** 18
    """ Target total sample count for adaptive ray batching. """

    mesh_log_every: int = 50
    """ Log SDF mesh to wandb every N epochs. Set to -1 to disable. """

    mesh_resolution: int = 256
    """ Resolution of the marching cubes grid for mesh extraction. """

    save_valid_imgs: bool = False
    """ Whether to save images when running validation. """


class NeuSTrainer(BaseTrainer):
    """Trainer for NeuS2-style neural surface reconstruction.

    Combines RGB rendering loss with Eikonal SDF regularization.
    Periodically extracts and logs meshes to wandb.
    """

    def __init__(self,
                 cfg: ConfigNeuSTrainer,
                 pipeline: wisp.models.Pipeline,
                 train_dataset: wisp.datasets.WispDataset,
                 validation_dataset: wisp.datasets.WispDataset,
                 tracker: Tracker,
                 device: torch.device = 'cuda',
                 scene_state: Optional[wisp.framework.WispState] = None):
        super().__init__(
            cfg=cfg, pipeline=pipeline, train_dataset=train_dataset,
            tracker=tracker, device=device, scene_state=scene_state,
        )
        self.validation_dataset = validation_dataset

    def populate_scenegraph(self):
        super().populate_scenegraph()
        self.scene_state.graph.cameras = self.train_dataset.cameras

    def pre_step(self):
        super().pre_step()
        # Progressive LOD activation
        self._update_active_lods()
        # Pruning
        if (self.cfg.prune_every > -1
                and self.total_iterations > 1
                and self.total_iterations % self.cfg.prune_every == 0):
            self.pipeline.nef.prune()

    def _update_active_lods(self):
        """Coarse-to-fine: gradually enable higher-resolution hash grid LODs."""
        nef = self.pipeline.nef
        num_lods = nef.grid.num_lods
        if self._progressive_lod_interval <= 0:
            nef.max_active_lods = num_lods
            return
        new_lods = min(1 + self.total_iterations // self._progressive_lod_interval, num_lods)
        if new_lods != nef.max_active_lods:
            nef.max_active_lods = new_lods
            log.info(f"Progressive LOD: {new_lods}/{num_lods} active")

    def calc_adaptive_rays(self, rays, warmup=False):
        if warmup:
            raymarch_results = self.pipeline.nef.grid.raymarch(
                rays,
                level=self.pipeline.nef.grid.active_lods[-1],
                num_samples=self.pipeline.tracer.num_steps,
                raymarch_type=self.pipeline.tracer.raymarch_type,
            )
            self.pipeline.tracer.prev_num_samples = raymarch_results.samples.shape[0]

        samples_per_ray = self.pipeline.tracer.get_prev_num_samples() / rays.shape[0]
        num_rays = self.cfg.target_sample_size / max(samples_per_ray, 1)
        num_rays = int(math.floor(min(num_rays, 2 ** 18)))
        if isinstance(self.train_dataset.transform, SampleRays):
            self.train_dataset.transform.set_num_samples(num_rays)
        else:
            raise Exception("SampleRays should be used as the transform for the dataset")

    @torch.cuda.nvtx.range("NeuSTrainer.step")
    def step(self, data):
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['rgb'].to(self.device).squeeze(0)

        if self.pipeline.tracer.get_prev_num_samples() is None:
            self.calc_adaptive_rays(rays, warmup=True)
            return

        self.optimizer.zero_grad()
        loss = 0

        if self.cfg.random_lod:
            population = list(range(self.pipeline.nef.grid.num_lods))
            weights = [2 ** i for i in population]
            weights = [w / sum(weights) for w in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            lod_idx = None

        # Forward: volume rendering
        rb = self.pipeline(rays=rays, lod_idx=lod_idx, channels=["rgb"])

        # RGB loss
        if self.cfg.rgb_loss_type == 'l1':
            rgb_loss = torch.abs(rb.rgb - img_gts)
        elif self.cfg.rgb_loss_type == 'l2':
            rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
        elif self.cfg.rgb_loss_type == 'huber':
            rgb_loss = F.smooth_l1_loss(rb.rgb, img_gts, reduction='none')
        else:
            raise NotImplementedError(f"Unknown rgb_loss_type: {self.cfg.rgb_loss_type}")

        if self.cfg.rgb_loss_denom == 'samples':
            rgb_loss = rgb_loss.sum() / self.pipeline.tracer.prev_num_samples
        else:
            rgb_loss = rgb_loss.mean()
        loss += self.cfg.rgb_lambda * rgb_loss

        # Eikonal loss
        eikonal_loss = self._compute_eikonal_loss(lod_idx)
        loss += self.cfg.eikonal_lambda * eikonal_loss

        # Metrics
        self.tracker.metrics.total_loss += loss.item()
        self.tracker.metrics.rgb_loss += rgb_loss.item()
        self.tracker.metrics.eikonal_loss += eikonal_loss.item()
        self.tracker.metrics.num_samples += 1

        # Backward
        with torch.cuda.nvtx.range("NeuSTrainer.backward"):
            if self.cfg.enable_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        self.calc_adaptive_rays(rays, warmup=False)
        if self.cfg.scheduler:
            self.scheduler.step()

    def _compute_eikonal_loss(self, lod_idx) -> torch.Tensor:
        """Compute Eikonal regularization: ||grad(SDF)|| = 1."""
        nef = self.pipeline.nef
        last_samples = self.pipeline.tracer.last_samples
        if last_samples is None or last_samples.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        # Subsample for efficiency
        max_pts = 4096
        if last_samples.shape[0] > max_pts:
            idx = torch.randperm(last_samples.shape[0], device=last_samples.device)[:max_pts]
            pts = last_samples[idx]
        else:
            pts = last_samples

        # Add random points in scene bounds for regularization
        rand_pts = torch.rand_like(pts) * 2.0 - 1.0
        eikonal_pts = torch.cat([pts, rand_pts], dim=0)

        def sdf_fn(x):
            if lod_idx is None:
                _lod = len(nef.grid.active_lods) - 1
            else:
                _lod = lod_idx
            dummy_dirs = torch.zeros_like(x)
            return nef.forward(coords=x, ray_d=dummy_dirs, lod_idx=_lod, channels="sdf")

        sdf_grads = finitediff_gradient(eikonal_pts, sdf_fn)
        eikonal_loss = ((sdf_grads.norm(dim=-1) - 1.0) ** 2).mean()
        return eikonal_loss

    def log_console(self):
        total_loss = self.tracker.metrics.average_metric('total_loss')
        rgb_loss = self.tracker.metrics.average_metric('rgb_loss')
        eikonal_loss = self.tracker.metrics.average_metric('eikonal_loss')
        inv_s = self.pipeline.nef.inv_s.item()
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log_text += ' | rgb loss: {:>.3E}'.format(rgb_loss)
        log_text += ' | eikonal: {:>.3E}'.format(eikonal_loss)
        log_text += ' | inv_s: {:.2f}'.format(inv_s)
        log.info(log_text)
        # Log inv_s directly to tracker (not through MetricsBoard)
        self.tracker.log_metric('inv_s', inv_s, self.epoch)

    def post_epoch(self):
        super().post_epoch()
        if (self.cfg.mesh_log_every > 0
                and self.epoch > 0
                and self.epoch % self.cfg.mesh_log_every == 0):
            self._log_mesh()

    def _log_mesh(self):
        """Extract SDF mesh and log to wandb."""
        if marching_cubes is None:
            log.warning("skimage not found, skipping mesh extraction.")
            return

        verts, faces = self.extract_mesh(self.cfg.mesh_resolution)
        if verts is None:
            return

        # Save OBJ locally
        mesh_dir = os.path.join(self.tracker.log_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        obj_path = os.path.join(mesh_dir, f"mesh_epoch{self.epoch:04d}.obj")
        self._save_obj(obj_path, verts, faces)

        # Log to wandb
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {"mesh": wandb.Object3D(open(obj_path))},
                step=self.epoch,
                commit=False,
            )
            log.info(f"Logged mesh to wandb (epoch {self.epoch})")

    @torch.no_grad()
    def extract_mesh(self, resolution: int = 256):
        """Extract mesh from SDF using marching cubes.

        Args:
            resolution: Grid resolution for evaluation.

        Returns:
            Tuple of (vertices, faces) numpy arrays, or (None, None) on failure.
        """
        self.pipeline.eval()
        nef = self.pipeline.nef
        device = next(nef.parameters()).device

        # Create uniform 3D grid in [-1, 1]
        x = torch.linspace(-1, 1, resolution)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(device)

        # Evaluate SDF in chunks to avoid OOM
        sdf_vals = []
        chunk_size = 2 ** 16
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords[i:i + chunk_size]
            dummy_dirs = torch.zeros_like(chunk)
            sdf = nef(coords=chunk, ray_d=dummy_dirs, channels="sdf")
            sdf_vals.append(sdf.cpu())
        sdf_vals = torch.cat(sdf_vals, dim=0)

        sdf_grid = sdf_vals.reshape(resolution, resolution, resolution).numpy()
        try:
            verts, faces, _, _ = marching_cubes(sdf_grid, level=0.0)
            # Scale vertices from grid coords to [-1, 1]
            verts = verts / (resolution - 1) * 2.0 - 1.0
        except ValueError:
            log.warning("Marching cubes failed (no zero crossing found).")
            return None, None

        self.pipeline.train()
        return verts, faces

    @staticmethod
    def _save_obj(path: str, verts: np.ndarray, faces: np.ndarray):
        with open(path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def pre_training(self):
        super().pre_training()
        self.tracker.metrics.define_metric('rgb_loss', aggregation_type=float)
        self.tracker.metrics.define_metric('eikonal_loss', aggregation_type=float)

        # Compute progressive LOD interval
        num_lods = self.pipeline.nef.grid.num_lods
        if self.cfg.progressive_lod_every > 0:
            self._progressive_lod_interval = self.cfg.progressive_lod_every
        else:
            # Auto: spread LOD activation across 80% of training
            total_iters = len(self.train_dataset) * self.cfg.max_epochs
            self._progressive_lod_interval = max(int(total_iters * 0.8) // num_lods, 1)
        log.info(f"Progressive LOD: activating 1 LOD every {self._progressive_lod_interval} iterations "
                 f"({num_lods} total LODs)")

    def validate(self):
        self.pipeline.eval()
        if self.validation_dataset is None:
            log.info("No validation dataset provided, skipping validation.")
            return

        img_shape = self.validation_dataset.img_shape
        imgs = list(self.validation_dataset.data["rgb"])
        rays = self.validation_dataset.data["rays"]
        ray_os = list(rays.origins.cuda())
        ray_ds = list(rays.dirs.cuda())
        imgs = [img.cuda() for img in imgs]

        total_psnr = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for idx, (img, ray_o, ray_d) in enumerate(zip(imgs, ray_os, ray_ds)):
                    _rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                    _rays = _rays.reshape(-1, 3)
                    rb = self.tracker.visualizer.render(self.pipeline, _rays, lod_idx=None)
                    rb = rb.reshape(*img_shape[:2], -1)
                    gts = img.reshape(*img_shape[:2], -1)
                    total_psnr += psnr(rb.rgb[..., :3], gts[..., :3])

        avg_psnr = total_psnr / len(imgs)
        log.info(f"EPOCH {self.epoch}/{self.max_epochs} | validation PSNR: {avg_psnr:.2f}")
        self.tracker.log_metric("validation/psnr", avg_psnr, self.epoch)
