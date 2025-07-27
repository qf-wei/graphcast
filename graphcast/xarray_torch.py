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
"""Helpers to use xarray.{Variable,DataArray,Dataset} with PyTorch.

Similar to xarray_jax.py but for PyTorch tensors. Allows them to be based on 
PyTorch tensors without converting to numpy arrays under the hood, so you can 
start with a PyTorch tensor, do some computation with it in xarray-land, get a 
PyTorch tensor out the other end and potentially use torch.jit through the 
whole thing.
"""

import collections
import contextlib
import contextvars
from typing import Any, Callable, Iterator, Mapping, Optional, Union, Tuple, TypeVar, cast
from typing import Hashable

import torch
import torch.nn.functional as F
import numpy as np
import tree
import xarray


_WRAPPED_TYPES = (torch.Tensor,)


def Variable(dims, data, **kwargs) -> xarray.Variable:
  """Like xarray.Variable, but can wrap PyTorch tensors."""
  return xarray.Variable(dims, wrap(data), **kwargs)


_TORCH_COORD_ATTR_NAME = '_torch_coord'


def DataArray(
    data,
    coords=None,
    dims=None,
    name=None,
    attrs=None,
    torch_coords=None,
    ) -> xarray.DataArray:
  """Like xarray.DataArray, but supports using PyTorch tensors.

  Args:
    data: As for xarray.DataArray, except torch tensors are also supported.
    coords: Coordinates for the array, see xarray.DataArray. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays. Their values will form a static part of the data structure.
    dims: See xarray.DataArray.
    name: See xarray.DataArray.
    attrs: See xarray.DataArray.
    torch_coords: Additional coordinates, which *can* use PyTorch tensors. These
      coordinates will be treated as PyTorch data, that means when using torch.jit
      they will be passed as tensors and computation involving them will be JIT'd.

  Returns:
    An instance of xarray.DataArray. Where PyTorch tensors are used as data or
    coords, they will be wrapped with TorchArrayWrapper and can be unwrapped via
    `unwrap` and `unwrap_data`.
  """
  result = xarray.DataArray(
      wrap(data), dims=dims, name=name, attrs=attrs or {})
  return assign_coords(result, coords=coords, torch_coords=torch_coords)


def Dataset(
    data_vars=None,
    coords=None,
    attrs=None,
    torch_coords=None,
    ) -> xarray.Dataset:
  """Like xarray.Dataset, but can wrap PyTorch tensors.

  Args:
    data_vars: As for xarray.Dataset, except torch tensors are also supported.
    coords: Coordinates for the dataset, see xarray.Dataset. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays.
    attrs: See xarray.Dataset.
    torch_coords: Additional coordinates, which *can* use PyTorch tensors.

  Returns:
    An instance of xarray.Dataset. Where PyTorch tensors are used as data, they
    will be wrapped with TorchArrayWrapper.
  """
  wrapped_data_vars = {}
  for name, var_like in (data_vars or {}).items():
    if isinstance(var_like, _WRAPPED_TYPES):
      wrapped_data_vars[name] = wrap(var_like)
    elif isinstance(var_like, tuple):
      wrapped_data_vars[name] = (var_like[0], wrap(var_like[1])) + var_like[2:]
    else:
      wrapped_data_vars[name] = var_like

  result = xarray.Dataset(
      data_vars=wrapped_data_vars,
      attrs=attrs)

  return assign_coords(result, coords=coords, torch_coords=torch_coords)


DatasetOrDataArray = TypeVar(
    'DatasetOrDataArray', xarray.Dataset, xarray.DataArray)


def assign_coords(
    x: DatasetOrDataArray,
    *,
    coords: Optional[Mapping[Hashable, Any]] = None,
    torch_coords: Optional[Mapping[Hashable, Any]] = None,
    ) -> DatasetOrDataArray:
  """Replacement for assign_coords which works in presence of torch_coords."""
  coords = {} if coords is None else dict(coords)
  torch_coords = {} if torch_coords is None else dict(torch_coords)

  existing_torch_coords = get_torch_coords(x)
  torch_coords = {**existing_torch_coords, **torch_coords}
  x = x.drop_vars(existing_torch_coords.keys())

  renamed_torch_coords = {}
  for name, coord in torch_coords.items():
    if isinstance(coord, xarray.DataArray):
      coord = coord.variable

    if isinstance(coord, list):
      coord = np.array(coord)

    if isinstance(coord, xarray.Variable):
      coord = coord.copy()
    elif isinstance(coord, tuple):
      dims, data = coord
      coord = Variable(dims, data)
    elif torch.is_tensor(coord) and coord.ndim == 0:
      coord = Variable(dims=(), data=coord)
    elif torch.is_tensor(coord) and coord.ndim == 1:
      coord = Variable((name,), coord)
    else:
      raise ValueError(f'Unsupported value for coordinate {name}')

    try:
      if hasattr(coord, 'attrs') and coord.attrs is not None:
        coord.attrs[_TORCH_COORD_ATTR_NAME] = True
    except (AttributeError, TypeError):
      pass
    renamed_torch_coords[f'__NONINDEX_{name}'] = coord

  x = x.assign_coords(coords={**coords, **renamed_torch_coords})

  rename_back_mapping = {f'__NONINDEX_{name}': name for name in torch_coords}
  if isinstance(x, xarray.Dataset):
    return x.rename_vars(rename_back_mapping)
  else:
    return x.rename(rename_back_mapping)


def get_torch_coords(x: DatasetOrDataArray) -> Mapping[Hashable, Any]:
  return {
      name: coord_var
      for name, coord_var in x.coords.variables.items()
      if coord_var.attrs.get(_TORCH_COORD_ATTR_NAME, False)}


def assign_torch_coords(
    x: DatasetOrDataArray,
    torch_coords: Optional[Mapping[Hashable, Any]] = None,
    **torch_coords_kwargs
    ) -> DatasetOrDataArray:
  """Assigns only torch_coords, with same API as xarray's assign_coords."""
  combined_coords = dict(torch_coords or {})
  for k, v in torch_coords_kwargs.items():
    combined_coords[k] = v
  return assign_coords(x, torch_coords=combined_coords)


def wrap(value):
  """Wraps PyTorch tensors for use in xarray, passing through other values."""
  if isinstance(value, _WRAPPED_TYPES):
    return TorchArrayWrapper(value)
  else:
    return value


def unwrap(value, require_torch=False):
  """Unwraps wrapped PyTorch tensors used in xarray, passing through other values."""
  if isinstance(value, TorchArrayWrapper):
    return value.torch_tensor
  elif isinstance(value, torch.Tensor):
    return value
  elif require_torch:
    raise TypeError(f'Expected PyTorch tensor, found {type(value)}.')
  else:
    return value


def _wrapped(func):
  """Surrounds a function with PyTorch tensor unwrapping/wrapping."""
  def wrapped_func(*args, **kwargs):
    args, kwargs = tree.map_structure(unwrap, (args, kwargs))
    result = func(*args, **kwargs)
    return tree.map_structure(wrap, result)
  return wrapped_func


def unwrap_data(
    value: Union[xarray.Variable, xarray.DataArray],
    require_torch: bool = False) -> Union[torch.Tensor, Any]:
  """Unwraps the data from an xarray Variable or DataArray."""
  return unwrap(value.data, require_torch=require_torch)


def unwrap_vars(
    dataset: xarray.Dataset,
    require_torch: bool = False) -> Mapping[Hashable, torch.Tensor]:
  """Unwraps all data variables from a Dataset."""
  return {name: unwrap_data(var, require_torch=require_torch)
          for name, var in dataset.data_vars.items()}


def unwrap_coords(
    dataset_or_data_array: DatasetOrDataArray,
    require_torch: bool = False) -> Mapping[Hashable, torch.Tensor]:
  """Unwraps all coordinate variables."""
  return {name: unwrap_data(coord, require_torch=require_torch)
          for name, coord in dataset_or_data_array.coords.items()}


def torch_data(
    value: Union[xarray.Variable, xarray.DataArray]) -> torch.Tensor:
  """Returns the data as a PyTorch tensor, erroring if it's not."""
  return unwrap_data(value, require_torch=True)


def torch_vars(dataset: xarray.Dataset) -> Mapping[Hashable, torch.Tensor]:
  """Returns all data variables as PyTorch tensors."""
  return unwrap_vars(dataset, require_torch=True)


class TorchArrayWrapper:
  """Wraps a PyTorch tensor to make it compatible with xarray operations."""

  def __init__(self, torch_tensor: torch.Tensor):
    if not isinstance(torch_tensor, torch.Tensor):
      raise TypeError(f"Expected torch.Tensor, got {type(torch_tensor)}")
    self.torch_tensor = torch_tensor

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if method != '__call__':
      return NotImplemented
    
    unwrapped_inputs = []
    for inp in inputs:
      if isinstance(inp, TorchArrayWrapper):
        unwrapped_inputs.append(inp.torch_tensor)
      elif isinstance(inp, torch.Tensor):
        unwrapped_inputs.append(inp)
      else:
        unwrapped_inputs.append(inp)
    
    try:
      result = ufunc(*unwrapped_inputs, **kwargs)
      if isinstance(result, torch.Tensor):
        return TorchArrayWrapper(result)
      return result
    except (TypeError, AttributeError):
      return NotImplemented

  def __array_function__(self, func, types, args, kwargs):
    return NotImplemented

  def __repr__(self):
    return f"TorchArrayWrapper({self.torch_tensor})"

  @property
  def shape(self):
    return tuple(self.torch_tensor.shape)

  @property
  def dtype(self):
    return self.torch_tensor.dtype

  @property
  def ndim(self):
    return self.torch_tensor.ndim

  @property
  def size(self):
    return self.torch_tensor.numel()

  @property
  def real(self):
    return TorchArrayWrapper(self.torch_tensor.real)

  @property
  def imag(self):
    return TorchArrayWrapper(self.torch_tensor.imag)

  def __array__(self, dtype=None):
    return self.torch_tensor.detach().cpu().numpy().astype(dtype)


def apply_ufunc(
    func: Callable,
    *args,
    input_core_dims=None,
    output_core_dims=None,
    exclude_dims=frozenset(),
    vectorize=False,
    join="exact",
    dataset_join="exact",
    dataset_fill_value=None,
    keep_attrs=None,
    kwargs=None,
    dask="forbidden",
    output_dtypes=None,
    output_sizes=None,
    meta=None,
    **kwargs_outer
):
  """PyTorch version of xarray.apply_ufunc."""
  
  @_wrapped
  def wrapped_func(*args, **kwargs):
    return func(*args, **kwargs)
  
  return xarray.apply_ufunc(
      wrapped_func, *args,
      input_core_dims=input_core_dims,
      output_core_dims=output_core_dims,
      exclude_dims=exclude_dims,
      vectorize=vectorize,
      join=join,
      dataset_join=dataset_join,
      dataset_fill_value=dataset_fill_value,
      keep_attrs=keep_attrs,
      kwargs=kwargs,
      dask=dask,
      output_dtypes=output_dtypes,
      output_sizes=output_sizes,
      meta=meta,
      **kwargs_outer
  )


def data_parallel(fn, dim, devices=None):
  """PyTorch equivalent of pmap for data parallel execution."""
  if devices is None:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [torch.device('cpu')]
  
  def result_fn(*args, **kwargs):
    if not args:
      return fn(**kwargs)
    
    first_arg = args[0]
    if not isinstance(first_arg, (xarray.Dataset, xarray.DataArray)):
      return fn(*args, **kwargs)
    
    if dim not in first_arg.dims:
      return fn(*args, **kwargs)
    
    dim_size = first_arg.sizes[dim]
    if len(devices) == 1 or dim_size == 1:
      return fn(*args, **kwargs)
    
    chunk_size = dim_size // len(devices)
    if chunk_size == 0:
      return fn(*args, **kwargs)
    
    results = []
    for i, device in enumerate(devices):
      start_idx = i * chunk_size
      end_idx = start_idx + chunk_size if i < len(devices) - 1 else dim_size
      
      chunk_args = []
      for arg in args:
        if isinstance(arg, (xarray.Dataset, xarray.DataArray)) and dim in arg.dims:
          chunk = arg.isel({dim: slice(start_idx, end_idx)})
          chunk_args.append(chunk)
        else:
          chunk_args.append(arg)
      
      chunk_result = fn(*chunk_args, **kwargs)
      results.append(chunk_result)
    
    return xarray.concat(results, dim=dim)
  
  return result_fn


def tree_map_variables(func, *args, **kwargs):
  """Apply a function to all Variables in xarray objects."""
  def map_var(var):
    if isinstance(var, xarray.Variable):
      return var._constructor(var.dims, func(var.data), var.attrs)
    return var
  
  return tree.map_structure(map_var, *args, **kwargs)


torch.fx.wrap('unwrap')
torch.fx.wrap('wrap')
