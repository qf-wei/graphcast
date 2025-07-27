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
"""PyTorch version of rollout utilities for GraphCast models."""

import torch
import xarray as xr
import numpy as np
from typing import Iterator, Optional, Sequence, Callable, Any
import logging


def chunked_prediction_generator_multiple_runs(
    predictor_fn: Callable,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    num_steps_per_chunk: int = 1,
    num_samples: int = 1,
    **kwargs
) -> Iterator[xr.Dataset]:
    """Generate chunked predictions for multiple ensemble runs."""
    
    for i in range(num_samples):
        logging.info(f"Sample {i+1}/{num_samples}")
        
        for prediction_chunk in chunked_prediction_generator(
            predictor_fn=predictor_fn,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=num_steps_per_chunk,
            **kwargs
        ):
            yield prediction_chunk.expand_dims('sample').assign_coords(sample=[i])


def chunked_prediction_generator(
    predictor_fn: Callable,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    num_steps_per_chunk: int = 1,
    **kwargs
) -> Iterator[xr.Dataset]:
    """Generate predictions in chunks for memory efficiency."""
    
    total_steps = targets_template.dims['time']
    current_inputs = inputs
    
    for start_step in range(0, total_steps, num_steps_per_chunk):
        end_step = min(start_step + num_steps_per_chunk, total_steps)
        
        chunk_targets_template = targets_template.isel(
            time=slice(start_step, end_step))
        chunk_forcings = forcings.isel(
            time=slice(start_step, end_step))
        
        chunk_predictions = predictor_fn(
            current_inputs,
            chunk_targets_template,
            chunk_forcings,
            **kwargs
        )
        
        yield chunk_predictions
        
        if end_step < total_steps:
            current_inputs = _get_next_inputs(current_inputs, chunk_predictions)


def _get_next_inputs(current_inputs: xr.Dataset, 
                    next_frame: xr.Dataset) -> xr.Dataset:
    """Update inputs for next prediction step."""
    num_inputs = current_inputs.dims['time']
    
    predicted_or_forced_inputs = next_frame[list(current_inputs.keys())]
    
    return (xr.concat([current_inputs, predicted_or_forced_inputs], dim='time')
            .tail(time=num_inputs)
            .assign_coords(time=current_inputs.coords['time']))


def extend_targets_template(targets_template: xr.Dataset,
                          num_additional_steps: int) -> xr.Dataset:
    """Extend targets template with additional time steps."""
    if num_additional_steps <= 0:
        return targets_template
    
    time_coord = targets_template.coords['time']
    time_delta = time_coord[1] - time_coord[0] if len(time_coord) > 1 else np.timedelta64(12, 'h')
    
    additional_times = [time_coord[-1] + (i + 1) * time_delta 
                       for i in range(num_additional_steps)]
    
    extended_time = np.concatenate([time_coord.values, additional_times])
    
    extended_template = targets_template.reindex(time=extended_time, method='nearest')
    
    return extended_template
