# Copyright 2024 DeepMind Technologies Limited.
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
"""Constructors for MLPs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNormConditioning(nn.Module):
  """Module for norm conditioning.

  Conditions the normalization of "inputs" by applying a linear layer to the
  "norm_conditioning" which produces the scale and variance which are applied to
  each channel (across the last dim) of "inputs".
  """

  def __init__(self, name="norm_conditioning"):
    super().__init__()
    self.name = name
    self.conditional_linear = None

  def forward(self, inputs: torch.Tensor, norm_conditioning: torch.Tensor):
    feature_size = inputs.shape[-1]
    
    if self.conditional_linear is None:
      self.conditional_linear = nn.Linear(
          norm_conditioning.shape[-1], 
          2 * feature_size
      ).to(inputs.device)
      nn.init.normal_(self.conditional_linear.weight, std=1e-8)
      nn.init.zeros_(self.conditional_linear.bias)
    
    conditional_scale_offset = self.conditional_linear(norm_conditioning)
    scale_minus_one, offset = torch.split(conditional_scale_offset, feature_size, dim=-1)
    scale = scale_minus_one + 1.0
    return inputs * scale + offset
