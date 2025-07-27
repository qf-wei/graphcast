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
"""PyTorch version of autoregressive predictor wrapper."""

import torch
import torch.nn as nn
import xarray as xr
from typing import Optional, Tuple, cast
import logging

from graphcast import xarray_torch
from graphcast import xarray_tree


class Predictor(nn.Module):
    """PyTorch version of autoregressive predictor wrapper."""
    
    def __init__(self,
                 predictor: nn.Module,
                 noise_level: Optional[float] = None,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.predictor = predictor
        self.noise_level = noise_level
        self.gradient_checkpointing = gradient_checkpointing
        
    def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
        constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
        constant_inputs = constant_inputs.drop_vars(
            forcings.keys(), errors='ignore')
        for name, var in constant_inputs.items():
            if 'time' in var.dims:
                raise ValueError(
                    f'Time-dependent input variable {name} must either be a forcing '
                    'variable, or a target variable to allow for auto-regressive '
                    'feedback.')
        return constant_inputs
    
    def _validate_targets_and_forcings(self, targets, forcings):
        for name, var in targets.items():
            if 'time' not in var.dims:
                raise ValueError(f'Target variable {name} must be time-dependent.')

        for name, var in forcings.items():
            if 'time' not in var.dims:
                raise ValueError(f'Forcing variable {name} must be time-dependent.')

        overlap = forcings.keys() & targets.keys()
        if overlap:
            raise ValueError('The following were specified as both targets and '
                           f'forcings, which isn\'t allowed: {overlap}')
    
    def _update_inputs(self, inputs, next_frame):
        num_inputs = inputs.dims['time']
        predicted_or_forced_inputs = next_frame[list(inputs.keys())]
        
        return (xr.concat([inputs, predicted_or_forced_inputs], dim='time')
                .tail(time=num_inputs)
                .assign_coords(time=inputs.coords['time']))
    
    def forward(self,
                inputs: xr.Dataset,
                targets_template: xr.Dataset,
                forcings: xr.Dataset,
                **kwargs) -> xr.Dataset:
        """Autoregressive forward pass."""
        constant_inputs = self._get_and_validate_constant_inputs(
            inputs, targets_template, forcings)
        self._validate_targets_and_forcings(targets_template, forcings)
        
        inputs = inputs.drop_vars(constant_inputs.keys())
        
        target_template = targets_template.isel(time=[0])
        
        predictions = []
        current_inputs = inputs
        
        for t in range(targets_template.dims['time']):
            current_forcings = forcings.isel(time=[t])
            current_forcings = current_forcings.assign_coords(
                time=target_template.coords['time'])
            
            all_inputs = xr.merge([constant_inputs, current_inputs])
            
            prediction = self.predictor(
                all_inputs, target_template,
                forcings=current_forcings,
                **kwargs)
            
            predictions.append(prediction.squeeze('time', drop=True))
            
            if t < targets_template.dims['time'] - 1:
                next_frame = xr.merge([prediction, current_forcings])
                current_inputs = self._update_inputs(current_inputs, next_frame)
        
        result = xr.concat(predictions, dim='time')
        
        for var_name in result.data_vars:
            target_var = targets_template[var_name]
            if result[var_name].dims != target_var.dims:
                result[var_name] = result[var_name].transpose(*target_var.dims)
        
        result = result.assign_coords(time=targets_template.coords['time'])
        return result
    
    def loss(self,
             inputs: xr.Dataset,
             targets: xr.Dataset,
             forcings: xr.Dataset,
             **kwargs) -> Tuple[torch.Tensor, xr.Dataset]:
        """Compute autoregressive loss."""
        if targets.sizes['time'] == 1:
            return self.predictor.loss(inputs, targets, forcings, **kwargs)
        
        constant_inputs = self._get_and_validate_constant_inputs(
            inputs, targets, forcings)
        self._validate_targets_and_forcings(targets, forcings)
        inputs = inputs.drop_vars(constant_inputs.keys())
        
        if self.noise_level:
            def add_noise(x):
                if isinstance(x, xr.DataArray):
                    data = xarray_torch.torch_data(x)
                    noise = torch.randn_like(data) * self.noise_level
                    return xarray_torch.DataArray(data + noise, dims=x.dims, coords=x.coords)
                return x
            inputs = xr.Dataset({k: add_noise(v) for k, v in inputs.items()})
        
        total_loss = None
        all_diagnostics = {}
        current_inputs = inputs
        
        for t in range(targets.dims['time']):
            current_target = targets.isel(time=[t])
            current_forcings = forcings.isel(time=[t])
            current_forcings = current_forcings.assign_coords(
                time=current_target.coords['time'])
            
            all_inputs = xr.merge([constant_inputs, current_inputs])
            
            if hasattr(self.predictor, 'loss_and_predictions'):
                (loss, diagnostics), prediction = self.predictor.loss_and_predictions(
                    all_inputs, current_target, forcings=current_forcings, **kwargs)
            else:
                loss, diagnostics = self.predictor.loss(
                    all_inputs, current_target, forcings=current_forcings, **kwargs)
                prediction = self.predictor(
                    all_inputs, current_target, forcings=current_forcings, **kwargs)
            
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
            
            for key, value in diagnostics.items():
                if key not in all_diagnostics:
                    all_diagnostics[key] = []
                all_diagnostics[key].append(value)
            
            if t < targets.dims['time'] - 1:
                next_frame = xr.merge([prediction, current_forcings])
                current_inputs = self._update_inputs(current_inputs, next_frame)
        
        avg_loss = total_loss / targets.dims['time']
        
        avg_diagnostics = {}
        for key, values in all_diagnostics.items():
            if values:
                stacked = xr.concat(values, dim='time')
                avg_diagnostics[key] = stacked.mean('time', skipna=False)
        
        return avg_loss, xr.Dataset(avg_diagnostics)
    
    def loss_and_predictions(self,
                           inputs: xr.Dataset,
                           targets: xr.Dataset,
                           forcings: xr.Dataset,
                           **kwargs) -> Tuple[Tuple[torch.Tensor, xr.Dataset], xr.Dataset]:
        """Compute loss and return predictions."""
        predictions = self.forward(inputs, targets, forcings, **kwargs)
        loss, diagnostics = self.loss(inputs, targets, forcings, **kwargs)
        return (loss, diagnostics), predictions
