# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modifications Copyright (c) 2026, NeuS2 tracer for kaolin-wisp.

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.tracers import BaseTracer
from typing import Tuple


class PackedNeuSTracer(BaseTracer):
    """Tracer for NeuS-style SDF volume rendering with packed ray samples.

    Replaces the NeRF density with SDF -> density conversion via the neural field's
    sdf_to_density() method. The rest of the volume rendering pipeline
    (exponential integration, alpha compositing) is identical to PackedRFTracer.
    """

    def __init__(self,
                 raymarch_type: str = 'ray',
                 num_steps: int = 512,
                 step_size: float = 1.0,
                 bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        super().__init__(bg_color=bg_color)
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)
        self.prev_num_samples = None
        self.last_samples = None  # Stored for Eikonal loss computation

    def get_prev_num_samples(self):
        return self.prev_num_samples

    def get_supported_channels(self):
        return {"depth", "hit", "rgb", "alpha"}

    def get_required_nef_channels(self):
        return {"rgb", "sdf"}

    def trace(self, nef, rays, channels, extra_channels,
              lod_idx=None, raymarch_type='ray', num_steps=512, step_size=1.0,
              bg_color='white'):
        """Trace rays through the NeuS neural field.

        Args:
            nef (nn.Module): A NeuralNeuSField that outputs SDF + RGB.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3].
            channels (set): Requested output channels.
            extra_channels (set): Additional channels for volumetric integration.
            lod_idx (int): LOD index to render at.
            raymarch_type (str): 'voxel' or 'ray'.
            num_steps (int): Number of samples per ray.
            step_size (float): Step size (unused, reserved).
            bg_color: Background color.

        Returns:
            RenderBuffer: Output buffers (rgb, alpha, depth, hit).
        """
        assert nef.grid is not None and "this tracer requires a grid"

        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        # Ray marching: generate samples along rays using occupancy structure
        raymarch_results = nef.grid.raymarch(
            rays,
            level=nef.grid.active_lods[lod_idx],
            num_samples=num_steps,
            raymarch_type=raymarch_type,
        )
        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        depths = raymarch_results.depth_samples
        boundary = raymarch_results.boundary

        num_samples = samples.shape[0]
        self.prev_num_samples = num_samples
        self.last_samples = samples.detach()  # Store for Eikonal loss

        hit_ray_d = rays.dirs.index_select(0, ridx)

        # Query neural field: SDF + RGB
        sdf, color = nef(
            coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=["sdf", "rgb"]
        )
        sdf = sdf.reshape(num_samples, 1)

        # NeuS core: convert SDF to density
        density = nef.sdf_to_density(sdf)

        # Volume rendering (identical to PackedRFTracer)
        self.bg_color = self.bg_color.to(rays.origins.device)

        depth = torch.zeros(N, 1, device=rays.origins.device) if "depth" in channels else None
        rgb = torch.zeros(N, 3, device=rays.origins.device) + self.bg_color
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)

        ridx_hit = ridx[boundary]

        tau = density * deltas
        del density, deltas
        ray_colors, transmittance = spc_render.exponential_integration(
            color, tau, boundary, exclusive=True
        )

        if depth is not None:
            ray_depth = spc_render.sum_reduce(
                depths.reshape(num_samples, 1) * transmittance, boundary
            )
            depth[ridx_hit, :] = ray_depth

        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[..., 0] > 0.0

        rgb[ridx_hit] = (self.bg_color * (1.0 - alpha)) + ray_colors

        # Extra channels
        extra_outputs = {}
        for channel in extra_channels:
            feats = nef(
                coords=samples, ray_d=hit_ray_d, lod_idx=lod_idx, channels=channel
            )
            num_ch = feats.shape[-1]
            ray_feats, transmittance = spc_render.exponential_integration(
                feats.view(num_samples, num_ch), tau, boundary, exclusive=True
            )
            composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_ch, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, **extra_outputs)
