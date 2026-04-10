# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modifications Copyright (c) 2026, NeuS2 entry point for kaolin-wisp.

import os
import logging
import torch
from typing import Optional
from pathlib import Path

# Load docker/.env if present (for WANDB_API_KEY, etc.)
_dotenv_path = Path(__file__).resolve().parents[2] / "docker" / ".env"
if _dotenv_path.exists():
    with open(_dotenv_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip("'\"")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

from wisp.app_utils import default_log_setup
from wisp.config import parse_config, configure, autoconfig, instantiate, print_config
from wisp.framework import WispState
from wisp.accelstructs import OctreeAS, AxisAlignedBBoxAS
from wisp.models.grids import OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.models.nefs.neural_neus import NeuralNeuSField
from wisp.models.pipeline import Pipeline
from wisp.tracers.packed_neus_tracer import PackedNeuSTracer
from wisp.datasets import NeRFSyntheticDataset, RTMVDataset, SampleRays
from wisp.trainers.neus_trainer import NeuSTrainer, ConfigNeuSTrainer
from wisp.trainers.tracker import Tracker, ConfigTracker

# FusedAdam -> AdamW fallback
try:
    from apex.optimizers import FusedAdam  # noqa: F401
except ImportError:
    logging.warning("apex not found, falling back to torch.optim.AdamW")
    import torch.optim
    torch.optim.FusedAdam = torch.optim.AdamW  # Register alias for config instantiation


@configure
class NeuS2AppConfig:
    """A script for training NeuS2: SDF-based neural surface reconstruction with hash grid backbone."""

    blas: autoconfig(OctreeAS.make_dense, OctreeAS.from_pointcloud, AxisAlignedBBoxAS)
    """ Bottom Level Acceleration structure for occupancy tracking and ray acceleration. """
    grid: autoconfig(OctreeGrid, HashGrid.from_geometric, TriplanarGrid, CodebookOctreeGrid)
    """ Feature grid used by the neural field. """
    nef: autoconfig(NeuralNeuSField)
    """ Neural field: SDF + view-dependent color with hash grid backbone. """
    tracer: autoconfig(PackedNeuSTracer)
    """ Tracer for NeuS-style SDF volume rendering. """
    dataset: autoconfig(NeRFSyntheticDataset, RTMVDataset)
    """ Multiview dataset. """
    dataset_transform: autoconfig(SampleRays)
    """ Dataset transforms for ray sampling. """
    trainer: ConfigNeuSTrainer
    """ NeuS2 trainer configuration. """
    tracker: ConfigTracker
    """ Experiment tracker for tensorboard, wandb, and visualization. """
    log_level: int = logging.INFO
    """ Global log level. """
    pretrained: Optional[str] = None
    """ Path to pretrained model. None creates a new model. """
    device: str = 'cuda'
    """ Device for optimization. """
    interactive: bool = os.environ.get('WISP_HEADLESS') != '1'
    """ Interactive mode with GUI. """


cfg = parse_config(NeuS2AppConfig, yaml_arg='--config')
device = torch.device(cfg.device)
default_log_setup(cfg.log_level)
if cfg.interactive:
    cfg.tracer.bg_color = (0.0, 0.0, 0.0)
    cfg.trainer.render_every = -1
    cfg.trainer.save_every = -1
    cfg.trainer.valid_every = -1
print_config(cfg)

# Dataset
dataset_transform = instantiate(cfg.dataset_transform)
train_dataset = instantiate(cfg.dataset, transform=dataset_transform)
validation_dataset = None
if cfg.trainer.valid_every > -1 or cfg.trainer.mode == 'validate':
    validation_dataset = train_dataset.create_split(split=cfg.trainer.valid_split, transform=None)

# Model
if cfg.pretrained and cfg.trainer.model_format == "full":
    pipeline = torch.load(cfg.pretrained)
else:
    pointcloud = train_dataset.as_pointcloud() if train_dataset.supports_depth() else None
    blas = instantiate(cfg.blas, pointcloud=pointcloud)
    grid = instantiate(cfg.grid, blas=blas)
    nef = instantiate(cfg.nef, grid=grid)
    tracer = instantiate(cfg.tracer)
    pipeline = Pipeline(nef=nef, tracer=tracer)
    if cfg.pretrained and cfg.trainer.model_format == "state_dict":
        pipeline.load_state_dict(torch.load(cfg.pretrained))

# Trainer
exp_name: str = cfg.trainer.exp_name
scene_state: WispState = WispState()
tracker = Tracker(cfg=cfg.tracker, exp_name=exp_name)
tracker.save_app_config(cfg)
trainer = NeuSTrainer(
    cfg=cfg.trainer,
    pipeline=pipeline,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    tracker=tracker,
    device=device,
    scene_state=scene_state,
)

# Run
if not cfg.interactive:
    logging.info("Running headless. For the app, set --interactive=True or $WISP_HEADLESS=0.")
    if cfg.trainer.mode == 'validate':
        trainer.validate()
    elif cfg.trainer.mode == 'train':
        trainer.train()
else:
    from wisp.renderer.app.optimization_app import OptimizationApp
    scene_state.renderer.device = trainer.device
    app = OptimizationApp(
        wisp_state=scene_state,
        trainer_step_func=trainer.iterate,
        experiment_name=exp_name,
    )
    app.run()
