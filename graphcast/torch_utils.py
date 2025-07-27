# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for PyTorch conversion of GraphCast models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Mapping, Any


def get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
  """Get PyTorch activation function by name."""
  if activation == "relu":
    return F.relu
  elif activation == "swish" or activation == "silu":
    return F.silu
  elif activation == "gelu":
    return F.gelu
  elif activation == "tanh":
    return torch.tanh
  elif activation == "sigmoid":
    return torch.sigmoid
  else:
    raise ValueError(f"Unknown activation: {activation}")


def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
  """PyTorch equivalent of jraph.segment_sum."""
  if data.dim() == 1:
    result = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
  else:
    result_shape = (num_segments,) + data.shape[1:]
    result = torch.zeros(result_shape, dtype=data.dtype, device=data.device)
  
  result.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
  return result


def segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
  """PyTorch equivalent of jraph.segment_mean."""
  sums = segment_sum(data, segment_ids, num_segments)
  counts = segment_sum(torch.ones_like(segment_ids, dtype=data.dtype), segment_ids, num_segments)
  counts = torch.clamp(counts, min=1.0)
  if data.dim() > 1:
    counts = counts.view(-1, *([1] * (data.dim() - 1)))
  return sums / counts


def get_aggregate_edges_for_nodes_fn(fn_name: str) -> Callable:
  """Get aggregation function by name."""
  if fn_name == "segment_sum":
    return segment_sum
  elif fn_name == "segment_mean":
    return segment_mean
  else:
    raise ValueError(f"Unknown aggregation function: {fn_name}")


class MLP(nn.Module):
  """Multi-layer perceptron with configurable activation and normalization."""
  
  def __init__(self,
               output_sizes: list[int],
               activation: str = "relu",
               activate_final: bool = False,
               use_layer_norm: bool = False,
               use_bias: bool = True,
               name: str = "mlp"):
    super().__init__()
    self.name = name
    self.output_sizes = output_sizes
    self.activation_fn = get_activation_fn(activation)
    self.activate_final = activate_final
    self.use_layer_norm = use_layer_norm
    
    self.layers = nn.ModuleList()
    self.layer_norms = nn.ModuleList() if use_layer_norm else []
    
    for i, output_size in enumerate(output_sizes):
      layer = nn.LazyLinear(output_size, bias=use_bias)
      self.layers.append(layer)
      
      if use_layer_norm:
        if i < len(output_sizes) - 1 or activate_final:
          self.layer_norms.append(nn.LayerNorm(output_size))
        else:
          self.layer_norms.append(None)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for i, layer in enumerate(self.layers):
      x = layer(x)
      
      is_final = (i == len(self.layers) - 1)
      should_activate = not is_final or self.activate_final
      
      if self.use_layer_norm and i < len(self.layer_norms) and self.layer_norms[i] is not None:
        x = self.layer_norms[i](x)
      
      if should_activate:
        x = self.activation_fn(x)
    
    return x


def make_mlp_with_norm_conditioning(
    output_sizes: list[int],
    activation: str = "relu",
    activate_final: bool = False,
    use_layer_norm: bool = False,
    use_norm_conditioning: bool = False,
    name: str = "mlp") -> nn.Module:
  """Create MLP with optional norm conditioning."""
  
  class MLPWithNormConditioning(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = MLP(
          output_sizes=output_sizes,
          activation=activation,
          activate_final=activate_final,
          use_layer_norm=use_layer_norm,
          name=name
      )
      if use_norm_conditioning:
        from . import mlp as mlp_module
        self.norm_conditioning = mlp_module.LinearNormConditioning()
      else:
        self.norm_conditioning = None
    
    def forward(self, x: torch.Tensor, norm_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
      x = self.mlp(x)
      if self.norm_conditioning is not None and norm_conditioning is not None:
        x = self.norm_conditioning(x, norm_conditioning)
      return x
  
  return MLPWithNormConditioning()


def variance_scaling_init(tensor: torch.Tensor, scale: float = 1.0, mode: str = "fan_in", distribution: str = "normal"):
  """PyTorch equivalent of Haiku's VarianceScaling initializer."""
  num_input = tensor.size(-1) if mode == "fan_in" else tensor.size(0)
  num_output = tensor.size(0) if mode == "fan_in" else tensor.size(-1)
  
  if mode == "fan_avg":
    num_input = (tensor.size(-1) + tensor.size(0)) / 2.0
  
  if distribution == "normal":
    std = (scale / num_input) ** 0.5
    nn.init.normal_(tensor, mean=0.0, std=std)
  elif distribution == "uniform":
    limit = (3.0 * scale / num_input) ** 0.5
    nn.init.uniform_(tensor, -limit, limit)
  else:
    raise ValueError(f"Unknown distribution: {distribution}")


def truncated_normal_init(tensor: torch.Tensor, stddev: float = 1.0):
  """PyTorch equivalent of Haiku's TruncatedNormal initializer."""
  nn.init.trunc_normal_(tensor, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)
