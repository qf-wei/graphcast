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
"""Deep Graph Neural Network - PyTorch Implementation.

This module contains the PyTorch implementation of a Deep Graph Neural Network.
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Union

from graphcast import mlp
from graphcast import typed_graph_torch
from graphcast import typed_graph_net_torch
from graphcast import torch_utils
import torch
import torch.nn as nn


NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = torch.Tensor

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, Mapping[str, SenderFeatures], Mapping[str, ReceiverFeatures],
     Globals],
    NodeFeatures]

# Signature:
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
GNUpdateGlobalFn = Callable[
    [Mapping[str, NodeFeatures], Mapping[str, EdgeFeatures], Globals],
    Globals]

# Signature:
GraphToGraphNetwork = Callable[[typed_graph_torch.TypedGraph], typed_graph_torch.TypedGraph]


class DeepTypedGraphNet(nn.Module):
  """Deep Graph Neural Network - PyTorch Implementation.

  PyTorch implementation of the Deep Graph Neural Network that works with 
  TypedGraphs with typed nodes and edges. It runs message passing steps in 
  the latent space with an encoder and decoder.
  """

  def __init__(
      self,
      *,
      node_latent_size: Mapping[str, int],
      edge_latent_size: Mapping[str, int],
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      num_processor_repetitions: int = 1,
      embed_nodes: bool = True,
      embed_edges: bool = True,
      node_output_size: Optional[Mapping[str, int]] = None,
      edge_output_size: Optional[Mapping[str, int]] = None,
      include_sent_messages_in_node_update: bool = False,
      use_layer_norm: bool = True,
      use_norm_conditioning: bool = False,
      activation: str = "relu",
      f32_aggregation: bool = False,
      aggregate_edges_for_nodes_fn: str = "segment_sum",
      aggregate_normalization: Optional[float] = None,
      name: str = "DeepTypedGraphNet",
  ):
    super().__init__()
    self.name = name
    self._node_latent_size = node_latent_size
    self._edge_latent_size = edge_latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._embed_nodes = embed_nodes
    self._embed_edges = embed_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = (
        include_sent_messages_in_node_update)
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning
    self._activation = activation
    self._f32_aggregation = f32_aggregation
    self._aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
    self._aggregate_normalization = aggregate_normalization
    
    self.embedder_network = None
    self.processor_networks = []
    self.decoder_network = None

  def forward(
      self,
      input_graph: typed_graph_torch.TypedGraph,
      global_norm_conditioning: Optional[torch.Tensor] = None,
  ) -> typed_graph_torch.TypedGraph:
    """Forward pass of the learnable dynamics model."""
    if self.embedder_network is None:
      self._build_networks(input_graph, global_norm_conditioning)

    # Embed input features (if applicable).
    latent_graph_0 = self._embed(input_graph, global_norm_conditioning)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0, global_norm_conditioning)

    # Compute outputs from the last latent graph.
    return self._output(latent_graph_m, global_norm_conditioning)

  def _build_networks(self, input_graph: typed_graph_torch.TypedGraph, 
                      global_norm_conditioning: Optional[torch.Tensor] = None):
    """Build the networks based on the input graph structure."""
    self.embedder_network = self._get_embedder_network()
    self.processor_networks = self._get_processor_networks()
    self.decoder_network = self._get_decoder_network()

  def _embed(self, input_graph: typed_graph_torch.TypedGraph,
             global_norm_conditioning: Optional[torch.Tensor] = None) -> typed_graph_torch.TypedGraph:
    """Embeds the input graph features into a latent representation."""
    if self.embedder_network is not None:
      return self.embedder_network(input_graph)
    return input_graph

  def _process(self, latent_graph_0: typed_graph_torch.TypedGraph,
               global_norm_conditioning: Optional[torch.Tensor] = None) -> typed_graph_torch.TypedGraph:
    """Processes the latent graph with several steps of message passing."""
    latent_graph_i = latent_graph_0
    for processor_network_i in self.processor_networks:
      latent_graph_i = processor_network_i(latent_graph_i)
    return latent_graph_i

  def _output(self, latent_graph_m: typed_graph_torch.TypedGraph,
              global_norm_conditioning: Optional[torch.Tensor] = None) -> typed_graph_torch.TypedGraph:
    """Computes the output of the model from the last latent graph."""
    if self.decoder_network is not None:
      return self.decoder_network(latent_graph_m)
    return latent_graph_m

  def _get_embedder_network(self) -> GraphToGraphNetwork:
    """Gets the embedder network."""
    if not (self._embed_nodes or self._embed_edges):
      return lambda x: x

    embed_node_fn = {}
    embed_edge_fn = {}

    if self._embed_nodes:
      for node_set_name, latent_size in self._node_latent_size.items():
        embed_node_fn[node_set_name] = self._make_embed_fn(
            latent_size=latent_size,
            name=f"embed_node_{node_set_name}")

    if self._embed_edges:
      for edge_set_name, latent_size in self._edge_latent_size.items():
        embed_edge_fn[edge_set_name] = self._make_embed_fn(
            latent_size=latent_size,
            name=f"embed_edge_{edge_set_name}")

    return typed_graph_net_torch.GraphMapFeatures(
        embed_node_fn=embed_node_fn if embed_node_fn else None,
        embed_edge_fn=embed_edge_fn if embed_edge_fn else None)

  def _get_processor_networks(self) -> Sequence[GraphToGraphNetwork]:
    """Gets the processor networks."""
    processor_networks = []
    for i in range(self._num_message_passing_steps):
      processor_networks.append(
          self._get_processor_network(name=f"processor_{i}"))
    return processor_networks

  def _get_processor_network(self, name: str) -> GraphToGraphNetwork:
    """Gets a processor network."""
    update_edge_fn = {}
    for edge_set_name, latent_size in self._edge_latent_size.items():
      update_edge_fn[edge_set_name] = self._make_mlp_with_norm_conditioning(
          latent_size=latent_size,
          name=f"{name}_edge_{edge_set_name}")

    update_node_fn = {}
    for node_set_name, latent_size in self._node_latent_size.items():
      update_node_fn[node_set_name] = self._make_mlp_with_norm_conditioning(
          latent_size=latent_size,
          name=f"{name}_node_{node_set_name}")

    graph_network = typed_graph_net_torch.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        aggregate_edges_for_nodes_fn=self._aggregate_edges_for_nodes_fn)

    if self._num_processor_repetitions == 1:
      return graph_network
    else:
      return self._repeat_graph_network(
          graph_network, self._num_processor_repetitions)

  def _get_decoder_network(self) -> GraphToGraphNetwork:
    """Gets the decoder network."""
    if not (self._node_output_size or self._edge_output_size):
      return lambda x: x

    decode_node_fn = {}
    decode_edge_fn = {}

    if self._node_output_size:
      for node_set_name, output_size in self._node_output_size.items():
        decode_node_fn[node_set_name] = self._make_mlp_with_norm_conditioning(
            latent_size=output_size,
            name=f"decode_node_{node_set_name}")

    if self._edge_output_size:
      for edge_set_name, output_size in self._edge_output_size.items():
        decode_edge_fn[edge_set_name] = self._make_mlp_with_norm_conditioning(
            latent_size=output_size,
            name=f"decode_edge_{edge_set_name}")

    return typed_graph_net_torch.GraphMapFeatures(
        embed_node_fn=decode_node_fn if decode_node_fn else None,
        embed_edge_fn=decode_edge_fn if decode_edge_fn else None)

  def _make_embed_fn(self, latent_size: int, name: str) -> Callable:
    """Makes an embedding function."""
    return nn.LazyLinear(latent_size)

  def _make_mlp_with_norm_conditioning(
      self, latent_size: int, name: str) -> Callable:
    """Makes an MLP with norm conditioning."""
    return torch_utils.make_mlp_with_norm_conditioning(
        output_sizes=[self._mlp_hidden_size] * self._mlp_num_hidden_layers + [latent_size],
        activation=self._activation,
        activate_final=False,
        use_layer_norm=self._use_layer_norm,
        use_norm_conditioning=self._use_norm_conditioning,
        name=name)

  def _repeat_graph_network(
      self, graph_network: GraphToGraphNetwork, num_repetitions: int
  ) -> GraphToGraphNetwork:
    """Repeats a graph network."""
    def repeated_graph_network(graph):
      for _ in range(num_repetitions):
        graph = graph_network(graph)
      return graph
    return repeated_graph_network


def _make_mlp_with_norm_conditioning(
    *,
    latent_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation: str,
    activate_final: bool,
    use_layer_norm: bool,
    use_norm_conditioning: bool,
    name: str,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor
]:
  """Makes an MLP with norm conditioning."""
  
  class MLPWithNormConditioning(nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = torch_utils.MLP(
          output_sizes=[hidden_size] * num_hidden_layers + [latent_size],
          activation=activation,
          activate_final=activate_final,
          use_layer_norm=use_layer_norm,
          name=name
      )
      if use_norm_conditioning:
        self.norm_conditioning = mlp.LinearNormConditioning()
      else:
        self.norm_conditioning = None

    def forward(self, edge_or_node_features: torch.Tensor,
                sent_features: torch.Tensor,
                received_features: torch.Tensor,
                global_features: torch.Tensor) -> torch.Tensor:
      net_input = torch.cat([
          edge_or_node_features, sent_features, received_features, global_features
      ], dim=-1)
      net = self.mlp(net_input)
      if self.norm_conditioning is not None:
        net = self.norm_conditioning(net, global_features)
      return net

  return MLPWithNormConditioning()


def _get_activation(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
  return torch_utils.get_activation_fn(activation)
