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
"""PyTorch version of normalization wrappers for GraphCast predictors."""

import torch
import torch.nn as nn
import xarray as xr
from typing import Optional, Tuple
import logging

from graphcast import xarray_torch
from graphcast import xarray_tree


def normalize(values: xr.Dataset,
              scales: xr.Dataset,
              locations: Optional[xr.Dataset] = None) -> xr.Dataset:
    """Normalize variables using the given scales and locations."""
    def normalize_array(array):
        if array.name is None:
            raise ValueError("Can't look up normalization constants because array has no name.")
        
        if locations is not None and array.name in locations:
            array = array - locations[array.name].astype(array.dtype)
        else:
            if locations is not None:
                logging.warning('No normalization location found for %s', array.name)
                
        if array.name in scales:
            array = array / scales[array.name].astype(array.dtype)
        else:
            logging.warning('No normalization scale found for %s', array.name)
            
        return array
    return xarray_tree.map_structure(normalize_array, values)


def unnormalize(values: xr.Dataset,
                scales: xr.Dataset,
                locations: Optional[xr.Dataset] = None) -> xr.Dataset:
    """Unnormalize variables using the given scales and locations."""
    def unnormalize_array(array):
        if array.name is None:
            raise ValueError("Can't look up normalization constants because array has no name.")
            
        if array.name in scales:
            array = array * scales[array.name].astype(array.dtype)
        else:
            logging.warning('No normalization scale found for %s', array.name)
            
        if locations is not None and array.name in locations:
            array = array + locations[array.name].astype(array.dtype)
        else:
            if locations is not None:
                logging.warning('No normalization location found for %s', array.name)
                
        return array
    return xarray_tree.map_structure(unnormalize_array, values)


class InputsAndResiduals(nn.Module):
    """PyTorch version of InputsAndResiduals normalization wrapper."""
    
    def __init__(self,
                 predictor: nn.Module,
                 stddev_by_level: xr.Dataset,
                 mean_by_level: xr.Dataset,
                 diffs_stddev_by_level: xr.Dataset):
        super().__init__()
        self.predictor = predictor
        self.scales = stddev_by_level
        self.locations = mean_by_level
        self.residual_scales = diffs_stddev_by_level
        self.residual_locations = None
        
    def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
        if norm_prediction.sizes.get('time') != 1:
            raise ValueError(
                'normalization.InputsAndResiduals only supports predicting a '
                'single timestep.')
        if norm_prediction.name in inputs:
            prediction = unnormalize(
                norm_prediction, self.residual_scales, self.residual_locations)
            last_input = inputs[norm_prediction.name].isel(time=-1)
            prediction = prediction + last_input
            return prediction
        else:
            return unnormalize(norm_prediction, self.scales, self.locations)
    
    def _subtract_input_and_normalize_target(self, inputs, target):
        if target.sizes.get('time') != 1:
            raise ValueError(
                'normalization.InputsAndResiduals only supports wrapping predictors'
                'that predict a single timestep.')
        if target.name in inputs:
            target_residual = target
            last_input = inputs[target.name].isel(time=-1)
            target_residual = target_residual - last_input
            return normalize(
                target_residual, self.residual_scales, self.residual_locations)
        else:
            return normalize(target, self.scales, self.locations)
        
    def forward(self, 
                inputs: xr.Dataset,
                targets_template: xr.Dataset,
                forcings: xr.Dataset,
                **kwargs) -> xr.Dataset:
        """Forward pass with normalization."""
        norm_inputs = normalize(inputs, self.scales, self.locations)
        norm_forcings = normalize(forcings, self.scales, self.locations)
        norm_predictions = self.predictor(
            norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
        return xarray_tree.map_structure(
            lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
            norm_predictions)
    
    def loss(self,
             inputs: xr.Dataset,
             targets: xr.Dataset,
             forcings: xr.Dataset,
             **kwargs) -> Tuple[torch.Tensor, xr.Dataset]:
        """Compute loss on normalized inputs and targets."""
        norm_inputs = normalize(inputs, self.scales, self.locations)
        norm_forcings = normalize(forcings, self.scales, self.locations)
        norm_target_residuals = xarray_tree.map_structure(
            lambda t: self._subtract_input_and_normalize_target(inputs, t),
            targets)
        return self.predictor.loss(
            norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
    
    def loss_and_predictions(self,
                           inputs: xr.Dataset,
                           targets: xr.Dataset,
                           forcings: xr.Dataset,
                           **kwargs) -> Tuple[Tuple[torch.Tensor, xr.Dataset], xr.Dataset]:
        """The loss computed on normalized data, with unnormalized predictions."""
        norm_inputs = normalize(inputs, self.scales, self.locations)
        norm_forcings = normalize(forcings, self.scales, self.locations)
        norm_target_residuals = xarray_tree.map_structure(
            lambda t: self._subtract_input_and_normalize_target(inputs, t),
            targets)
        (loss, scalars), norm_predictions = self.predictor.loss_and_predictions(
            norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
        predictions = xarray_tree.map_structure(
            lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
            norm_predictions)
        return (loss, scalars), predictions
