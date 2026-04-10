# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modifications Copyright (c) 2026, NeuS2 implementation for kaolin-wisp.

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid, TriplanarGrid


class NeuralNeuSField(BaseNeuralField):
    """Neural field for NeuS-style volume rendering with SDF and view-dependent color.

    Maps 3D coordinates + view direction -> SDF + RGB.
    Uses a hash grid backbone for fast training (NeuS2 approach).
    The SDF is converted to density via a learnable sigmoid transformation.
    """

    def __init__(self,
                 grid: BLASGrid,
                 # embedder args
                 pos_embedder: str = 'none',
                 view_embedder: str = 'positional',
                 pos_multires: int = 10,
                 view_multires: int = 4,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'linear',
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 bias: bool = True,
                 # NeuS args
                 init_inv_s: float = 0.3,
                 # pruning args
                 prune_density_decay: Optional[float] = 0.6,
                 prune_min_density: Optional[float] = 0.01,
                 ):
        super().__init__()
        self.grid = grid

        # Embedders
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(
            pos_embedder, pos_multires, include_input=position_input
        )
        self.view_embedder, self.view_embed_dim = self.init_embedder(
            view_embedder, view_multires, include_input=True
        )

        # Decoders
        self.hidden_dim = hidden_dim
        self.decoder_sdf, self.decoder_color = self.init_decoders(
            activation_type, layer_type, num_layers, hidden_dim, bias
        )

        # Learnable inv_s: controls surface sharpness
        self._ln_inv_s = nn.Parameter(torch.tensor(np.log(init_inv_s) / 10.0, dtype=torch.float32))

        # Progressive training: start with coarse LODs, gradually enable finer ones
        self.max_active_lods = 1  # Updated by trainer

        # Pruning
        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        if embedder_type == 'none' and not include_input:
            return None, 0
        elif embedder_type == 'identity' or (embedder_type == 'none' and include_input):
            return nn.Identity(), 3
        elif embedder_type == 'positional':
            return get_positional_embedder(frequencies=frequencies, include_input=include_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type: {embedder_type}')

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim, bias):
        activation = get_activation_class(activation_type)
        layer = get_layer_class(layer_type)

        # SDF decoder: grid features -> [sdf (1) + geo_features (15)]
        decoder_sdf = BasicDecoder(
            input_dim=self.sdf_net_input_dim(),
            output_dim=16,
            activation=activation,
            bias=bias,
            layer=layer,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            skip=[],
        )

        # Color decoder: geo_features + view_embed -> RGB (3)
        decoder_color = BasicDecoder(
            input_dim=self.color_net_input_dim(),
            output_dim=3,
            activation=activation,
            bias=bias,
            layer=layer,
            num_layers=num_layers + 1,
            hidden_dim=hidden_dim,
            skip=[],
        )
        return decoder_sdf, decoder_color

    @property
    def inv_s(self) -> torch.Tensor:
        """Learnable inverse-s parameter for SDF->density conversion.
        Increases during training -> sharper surface.
        """
        return torch.exp(self._ln_inv_s * 10.0)

    def sdf_to_density(self, sdf: torch.Tensor) -> torch.Tensor:
        """Convert SDF values to density using Laplacian density (VolSDF-style).

        density = (inv_s / 2) * exp(-inv_s * |sdf|)

        Peaked at the zero-crossing (surface) and decays symmetrically on both sides.

        Args:
            sdf (torch.FloatTensor): SDF values of shape [..., 1]

        Returns:
            torch.FloatTensor: Density values of shape [..., 1]
        """
        inv_s = self.inv_s
        return (inv_s / 2.0) * torch.exp(-inv_s * torch.abs(sdf))

    def prune(self):
        if self.prune_density_decay is None or self.prune_min_density is None:
            return
        if self.grid is None:
            return
        if not isinstance(self.grid, (HashGrid, TriplanarGrid)):
            raise NotImplementedError(f'Pruning not implemented for {self.grid.__class__.__name__}')

        self.grid.occupancy = self.grid.occupancy.cuda()
        self.grid.occupancy = self.grid.occupancy * self.prune_density_decay
        points = self.grid.dense_points.cuda()
        res = 2.0 ** self.grid.blas.max_level
        samples = torch.rand(points.shape[0], 3, device=points.device)
        samples = points.float() + samples
        samples = samples / res
        samples = samples * 2.0 - 1.0
        sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
        with torch.no_grad():
            sdf = self.forward(coords=samples, ray_d=sample_views, channels="sdf")
            density = self.sdf_to_density(sdf)
        self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]
        mask = self.grid.occupancy > self.prune_min_density
        _points = points[mask]
        if _points.shape[0] == 0:
            return
        if hasattr(self.grid.blas.__class__, "from_quantized_points"):
            self.grid.blas = self.grid.blas.__class__.from_quantized_points(
                _points, self.grid.blas.max_level
            )
        else:
            raise Exception(
                f"The BLAS {self.grid.blas.__class__.__name__} does not support "
                "from_quantized_points, required for pruning."
            )

    def register_forward_functions(self):
        self._register_forward_function(self.sdf_rgb, ["sdf", "rgb"])

    def sdf_rgb(self, coords, ray_d, lod_idx=None):
        """Compute SDF and view-dependent color for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"sdf": torch.FloatTensor, "rgb": torch.FloatTensor}:
                - SDF tensor of shape [batch, 1]
                - RGB tensor of shape [batch, 3]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Grid feature interpolation
        feats = self.grid.interpolate(coords, lod_idx).reshape(batch, self.effective_feature_dim())

        # Progressive training: zero out features from inactive LODs
        if self.max_active_lods < self.grid.num_lods:
            feat_per_lod = self.grid.feature_dim
            active_dim = self.max_active_lods * feat_per_lod
            feats = feats.clone()
            feats[..., active_dim:] = 0.0

        # Optional position embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(batch, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # SDF decoder -> [sdf, geo_features]
        sdf_out = self.decoder_sdf(feats)
        sdf = sdf_out[..., 0:1]
        geo_feats = sdf_out[..., 1:]

        # View direction embedding + color decoder
        if self.view_embedder is not None:
            embedded_dir = self.view_embedder(ray_d).view(batch, self.view_embed_dim)
            color_input = torch.cat([geo_feats, embedded_dir], dim=-1)
        else:
            color_input = geo_feats
        rgb = torch.sigmoid(self.decoder_color(color_input))

        return dict(sdf=sdf, rgb=rgb)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            return self.grid.feature_dim * self.grid.num_lods
        return self.grid.feature_dim

    def sdf_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 15 + self.view_embed_dim  # 15 geo features from sdf decoder

    def public_properties(self) -> Dict[str, Any]:
        properties = {
            "grid": self.grid,
            "inv_s": self.inv_s.item(),
        }
        return properties
