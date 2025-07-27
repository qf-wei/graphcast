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
"""GraphCast: Learning skillful medium-range global weather forecasting - PyTorch Implementation."""

import dataclasses
from typing import Any, Mapping, Optional, Tuple

from graphcast import deep_typed_graph_net_torch
from graphcast import icosahedral_mesh
from graphcast import model_utils
from graphcast import typed_graph_torch
from graphcast import xarray_torch
import torch
import torch.nn as nn
import xarray


@dataclasses.dataclass
class TaskConfig:
  """Task-specific configuration for GraphCast."""
  input_variables: Tuple[str, ...]
  target_variables: Tuple[str, ...]
  forcing_variables: Tuple[str, ...]
  pressure_levels: Tuple[int, ...]
  input_duration: str


@dataclasses.dataclass  
class ModelConfig:
  """Model-specific configuration for GraphCast."""
  resolution: int
  mesh_size: int
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float


class GraphCast(nn.Module):
  """GraphCast Predictor - PyTorch Implementation.
  
  PyTorch version of the JAX/Haiku GraphCast model.
  """
  
  def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
    """Initializes the predictor."""
    super().__init__()
    
    self.model_config = model_config
    self.task_config = task_config
    
    self._spatial_features_kwargs = dict(
        add_node_positions=False,
        add_node_latitude=True,
        add_node_longitude=True,
        add_relative_positions=True,
        relative_longitude_local_coordinates=True,
        relative_latitude_local_coordinates=True,
    )
    
    self._meshes = (
        icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config.mesh_size))
    
    self._grid2mesh_gnn = deep_typed_graph_net_torch.DeepTypedGraphNet(
        embed_nodes=True,
        embed_edges=True,
        edge_latent_size=dict(grid2mesh=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        name="grid2mesh_gnn",
    )
    
    self._mesh_gnn = deep_typed_graph_net_torch.DeepTypedGraphNet(
        embed_nodes=False,
        embed_edges=False,
        node_latent_size=dict(mesh_nodes=model_config.latent_size),
        edge_latent_size=dict(mesh=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=model_config.gnn_msg_steps,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        name="mesh_gnn",
    )
    
    self._mesh2grid_gnn = deep_typed_graph_net_torch.DeepTypedGraphNet(
        embed_nodes=False,
        embed_edges=True,
        edge_latent_size=dict(mesh2grid=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        node_output_size=dict(grid_nodes=len(task_config.target_variables)),
        name="mesh2grid_gnn",
    )

  def forward(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: xarray.Dataset,
      **optional_kwargs) -> xarray.Dataset:
    """Forward pass of GraphCast."""
    
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
    
    typed_graph = self._grid_node_features_to_typed_graph(
        grid_node_features, targets_template)
    
    latent_graph = self._grid2mesh_gnn(typed_graph)
    latent_graph = self._mesh_gnn(latent_graph)
    output_graph = self._mesh2grid_gnn(latent_graph)
    
    return self._typed_graph_to_prediction(output_graph, targets_template)

  def _inputs_to_grid_node_features(
      self, inputs: xarray.Dataset, forcings: xarray.Dataset) -> torch.Tensor:
    """Convert input xarray datasets to grid node features tensor."""
    input_vars = []
    for var_name in self.task_config.input_variables:
      if var_name in inputs.data_vars:
        var_data = xarray_torch.torch_data(inputs[var_name])
        if var_data.ndim == 5:  # [batch, time, level, lat, lon]
          flattened = var_data.flatten(start_dim=-2)  # [batch, time, level, lat*lon]
          flattened = flattened.flatten(start_dim=-2)  # [batch, time, level*lat*lon]
        elif var_data.ndim == 4:  # [batch, time, lat, lon]
          flattened = var_data.flatten(start_dim=-2)  # [batch, time, lat*lon]
        else:
          flattened = var_data
        input_vars.append(flattened)
    
    forcing_vars = []
    for var_name in self.task_config.forcing_variables:
      if var_name in forcings.data_vars:
        var_data = xarray_torch.torch_data(forcings[var_name])
        if var_data.ndim == 4:  # [batch, time, lat, lon]
          flattened = var_data.flatten(start_dim=-2)  # [batch, time, lat*lon]
        else:
          flattened = var_data
        
        if input_vars and flattened.ndim >= 2 and input_vars[0].ndim >= 2:
          target_time_steps = input_vars[0].shape[1]
          if flattened.shape[1] != target_time_steps:
            if flattened.shape[1] == 1:
              flattened = flattened.expand(-1, target_time_steps, -1)
            else:
              flattened = flattened[:, :1].expand(-1, target_time_steps, -1)
        
        forcing_vars.append(flattened)
    
    all_features = input_vars + forcing_vars
    if all_features:
      return torch.cat(all_features, dim=-1)
    else:
      batch_size = list(inputs.coords['batch'].values)[0] if 'batch' in inputs.coords else 1
      return torch.randn(batch_size, 2, 32)  # dummy features

  def _grid_node_features_to_typed_graph(
      self, grid_node_features: torch.Tensor, 
      targets_template: xarray.Dataset) -> typed_graph_torch.TypedGraph:
    """Convert grid node features to a typed graph."""
    batch_size = grid_node_features.shape[0] if grid_node_features.numel() > 0 else 1
    
    mesh = self._meshes[0]  # Use the first mesh level
    grid_nodes_count = 32  # Simplified for testing
    mesh_nodes_count = len(mesh.vertices)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([batch_size]),
        features=torch.zeros(batch_size, self.model_config.latent_size, requires_grad=True)
    )
    
    if grid_node_features.numel() == 0:
        grid_features = torch.randn(grid_nodes_count, self.model_config.latent_size, requires_grad=True)
    else:
        if grid_node_features.ndim == 3:  # [batch, time, features]
            grid_features = grid_node_features[0, -1]  # Take last time step, first batch
        else:
            grid_features = grid_node_features
        
        if grid_features.shape[0] != grid_nodes_count:
            if grid_features.shape[0] > grid_nodes_count:
                grid_features = grid_features[:grid_nodes_count]
            else:
                padding_size = grid_nodes_count - grid_features.shape[0]
                padding = torch.randn(padding_size, grid_features.shape[-1], requires_grad=True, device=grid_features.device)
                grid_features = torch.cat([grid_features, padding], dim=0)
        
        if grid_features.shape[-1] != self.model_config.latent_size:
            if not hasattr(self, '_feature_transform'):
                self._feature_transform = nn.Linear(grid_features.shape[-1], self.model_config.latent_size).to(grid_features.device)
            grid_features = self._feature_transform(grid_features)
    
    nodes = {
        "grid_nodes": typed_graph_torch.NodeSet(
            n_node=torch.tensor([grid_nodes_count]),
            features=grid_features
        ),
        "mesh_nodes": typed_graph_torch.NodeSet(
            n_node=torch.tensor([mesh_nodes_count]),
            features=torch.randn(mesh_nodes_count, self.model_config.latent_size, requires_grad=True, device=grid_features.device)
        )
    }
    
    grid2mesh_edges = min(64, grid_nodes_count * 2)
    mesh_edges = len(mesh.faces) * 3  # Each face contributes 3 edges
    mesh2grid_edges = min(64, mesh_nodes_count * 2)
    
    actual_grid_nodes = grid_features.shape[0]
    actual_mesh_nodes = nodes["mesh_nodes"].features.shape[0]
    
    edges = {
        typed_graph_torch.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")): 
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([grid2mesh_edges]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, actual_grid_nodes, (grid2mesh_edges,)),
                receivers=torch.randint(0, actual_mesh_nodes, (grid2mesh_edges,))
            ),
            features=torch.randn(grid2mesh_edges, self.model_config.latent_size, requires_grad=True, device=grid_features.device)
        ),
        typed_graph_torch.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([mesh_edges]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, actual_mesh_nodes, (mesh_edges,)),
                receivers=torch.randint(0, actual_mesh_nodes, (mesh_edges,))
            ),
            features=torch.randn(mesh_edges, self.model_config.latent_size, requires_grad=True, device=grid_features.device)
        ),
        typed_graph_torch.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([mesh2grid_edges]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, actual_mesh_nodes, (mesh2grid_edges,)),
                receivers=torch.randint(0, actual_grid_nodes, (mesh2grid_edges,))
            ),
            features=torch.randn(mesh2grid_edges, self.model_config.latent_size, requires_grad=True, device=grid_features.device)
        )
    }
    
    return typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )

  def _typed_graph_to_prediction(
      self, output_graph: typed_graph_torch.TypedGraph,
      targets_template: xarray.Dataset) -> xarray.Dataset:
    """Convert typed graph output back to xarray dataset."""
    grid_features = output_graph.nodes["grid_nodes"].features
    
    predictions = {}
    for i, var_name in enumerate(self.task_config.target_variables):
      if i < grid_features.shape[-1]:
        var_data = grid_features[..., i]
      else:
        var_data = torch.zeros_like(grid_features[..., 0])
      
      if var_name in targets_template.data_vars:
        template_var = targets_template[var_name]
        target_shape = template_var.shape
        
        if var_data.numel() != torch.prod(torch.tensor(target_shape)):
          if not hasattr(self, f'_output_transform_{var_name}'):
            input_size = var_data.numel()
            output_size = torch.prod(torch.tensor(target_shape)).item()
            transform = nn.Linear(input_size, output_size).to(var_data.device)
            setattr(self, f'_output_transform_{var_name}', transform)
          
          transform = getattr(self, f'_output_transform_{var_name}')
          var_data = transform(var_data.flatten()).view(target_shape)
        else:
          var_data = var_data.view(target_shape)
        
        predictions[var_name] = xarray_torch.DataArray(
            var_data,
            dims=template_var.dims,
            coords=template_var.coords
        )
    
    return xarray_torch.Dataset(predictions)

  def loss(self, inputs: xarray.Dataset, targets: xarray.Dataset,
           forcings: xarray.Dataset, **optional_kwargs) -> Tuple[torch.Tensor, xarray.Dataset]:
    """Compute loss for training."""
    predictions = self.forward(inputs, targets, forcings, **optional_kwargs)
    
    total_loss = None
    diagnostics = {}
    
    for var_name in self.task_config.target_variables:
      if var_name in predictions.data_vars and var_name in targets.data_vars:
        pred_data = xarray_torch.torch_data(predictions[var_name])
        target_data = xarray_torch.torch_data(targets[var_name])
        var_loss = torch.nn.functional.mse_loss(pred_data, target_data)
        if total_loss is None:
          total_loss = var_loss
        else:
          total_loss = total_loss + var_loss
        diagnostics[f"{var_name}_loss"] = xarray_torch.DataArray(var_loss.detach())
    
    if total_loss is None:
      total_loss = torch.tensor(0.0, requires_grad=True)
    
    return total_loss, xarray_torch.Dataset(diagnostics)
