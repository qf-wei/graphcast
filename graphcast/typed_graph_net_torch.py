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
"""A library of typed Graph Neural Networks using PyTorch."""

from typing import Callable, Mapping, Optional, Union
import torch
import torch.nn as nn

from . import typed_graph_torch as typed_graph
from . import torch_utils


NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = torch.Tensor

GNUpdateNodeFn = Callable[
    [NodeFeatures, Mapping[str, SenderFeatures], Mapping[str, ReceiverFeatures], Globals],
    NodeFeatures]

GNUpdateGlobalFn = Callable[
    [Mapping[str, NodeFeatures], Mapping[str, EdgeFeatures], Globals],
    Globals]


class GraphNetwork(nn.Module):
  """PyTorch implementation of GraphNetwork for TypedGraphs."""
  
  def __init__(self,
               update_edge_fn: Mapping[str, Callable],
               update_node_fn: Mapping[str, GNUpdateNodeFn],
               update_global_fn: Optional[GNUpdateGlobalFn] = None,
               aggregate_edges_for_nodes_fn: str = "segment_sum",
               aggregate_nodes_for_globals_fn: str = "segment_sum",
               aggregate_edges_for_globals_fn: str = "segment_sum"):
    super().__init__()
    
    self.update_edge_fn = nn.ModuleDict({k: v for k, v in update_edge_fn.items()})
    self.update_node_fn = nn.ModuleDict({k: v for k, v in update_node_fn.items()})
    self.update_global_fn = update_global_fn
    
    self.aggregate_edges_for_nodes_fn = torch_utils.get_aggregate_edges_for_nodes_fn(
        aggregate_edges_for_nodes_fn)
    self.aggregate_nodes_for_globals_fn = torch_utils.get_aggregate_edges_for_nodes_fn(
        aggregate_nodes_for_globals_fn)
    self.aggregate_edges_for_globals_fn = torch_utils.get_aggregate_edges_for_nodes_fn(
        aggregate_edges_for_globals_fn)
  
  def forward(self, graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    """Applies the configured GraphNetwork to a graph."""
    updated_graph = graph

    updated_edges = dict(updated_graph.edges)
    for edge_set_name, edge_fn in self.update_edge_fn.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      updated_edges[edge_set_key] = self._edge_update(
          updated_graph, edge_fn, edge_set_key)
    updated_graph = updated_graph._replace(edges=updated_edges)

    updated_nodes = dict(updated_graph.nodes)
    for node_set_key, node_fn in self.update_node_fn.items():
      updated_nodes[node_set_key] = self._node_update(
          updated_graph, node_fn, node_set_key, self.aggregate_edges_for_nodes_fn)
    updated_graph = updated_graph._replace(nodes=updated_nodes)

    if self.update_global_fn:
      updated_context = self._global_update(
          updated_graph, self.update_global_fn,
          self.aggregate_edges_for_globals_fn,
          self.aggregate_nodes_for_globals_fn)
      updated_graph = updated_graph._replace(context=updated_context)

    return updated_graph

  def _edge_update(self, graph, edge_fn, edge_set_key):
    """Updates an edge set of a given key."""
    sender_nodes = graph.nodes[edge_set_key.node_sets[0]]
    receiver_nodes = graph.nodes[edge_set_key.node_sets[1]]
    edge_set = graph.edges[edge_set_key]
    senders = edge_set.indices.senders
    receivers = edge_set.indices.receivers

    def gather_features(features, indices):
      if isinstance(features, dict):
        return {k: v[indices] for k, v in features.items()}
      else:
        return features[indices]

    sent_attributes = gather_features(sender_nodes.features, senders)
    received_attributes = gather_features(receiver_nodes.features, receivers)

    n_edge = edge_set.n_edge
    sum_n_edge = senders.shape[0]
    
    def repeat_global_features(features):
      if isinstance(features, dict):
        return {k: v.unsqueeze(0).expand(sum_n_edge, -1) 
                for k, v in features.items()}
      else:
        if features.ndim == 1:
          return features.unsqueeze(0).expand(sum_n_edge, -1)
        elif features.ndim == 2:
          batch_size = features.shape[0]
          feature_dim = features.shape[1]
          edges_per_batch = sum_n_edge // batch_size
          return features.unsqueeze(1).expand(batch_size, edges_per_batch, feature_dim).reshape(sum_n_edge, feature_dim)
        else:
          return features.repeat_interleave(n_edge, dim=0)[:sum_n_edge]
    
    global_features = repeat_global_features(graph.context.features)
    new_features = edge_fn(edge_set.features, sent_attributes, received_attributes, global_features)
    return edge_set._replace(features=new_features)

  def _node_update(self, graph, node_fn, node_set_key, aggregation_fn):
    """Updates a node set of a given key."""
    node_set = graph.nodes[node_set_key]
    
    if isinstance(node_set.features, dict):
      sum_n_node = next(iter(node_set.features.values())).shape[0]
    else:
      sum_n_node = node_set.features.shape[0]

    sent_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      sender_node_set_key = edge_set_key.node_sets[0]
      if sender_node_set_key == node_set_key:
        senders = edge_set.indices.senders
        if isinstance(edge_set.features, dict):
          sent_features[edge_set_key.name] = {
              k: aggregation_fn(v, senders, sum_n_node) 
              for k, v in edge_set.features.items()
          }
        else:
          sent_features[edge_set_key.name] = aggregation_fn(
              edge_set.features, senders, sum_n_node)

    received_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      receiver_node_set_key = edge_set_key.node_sets[1]
      if receiver_node_set_key == node_set_key:
        receivers = edge_set.indices.receivers
        if isinstance(edge_set.features, dict):
          received_features[edge_set_key.name] = {
              k: aggregation_fn(v, receivers, sum_n_node) 
              for k, v in edge_set.features.items()
          }
        else:
          received_features[edge_set_key.name] = aggregation_fn(
              edge_set.features, receivers, sum_n_node)

    n_node = node_set.n_node
    
    def repeat_global_features(features):
      if isinstance(features, dict):
        return {k: v.unsqueeze(0).expand(sum_n_node, -1) 
                for k, v in features.items()}
      else:
        if features.ndim == 1:
          return features.unsqueeze(0).expand(sum_n_node, -1)
        elif features.ndim == 2:
          batch_size = features.shape[0]
          feature_dim = features.shape[1]
          nodes_per_batch = sum_n_node // batch_size
          return features.unsqueeze(1).expand(batch_size, nodes_per_batch, feature_dim).reshape(sum_n_node, feature_dim)
        else:
          return features.repeat_interleave(n_node, dim=0)[:sum_n_node]
    
    global_features = repeat_global_features(graph.context.features)
    new_features = node_fn(node_set.features, sent_features, received_features, global_features)
    return node_set._replace(features=new_features)

  def _global_update(self, graph, global_fn, edge_aggregation_fn, node_aggregation_fn):
    """Updates the global features."""
    n_graph = graph.context.n_graph.shape[0]
    graph_idx = torch.arange(n_graph, device=graph.context.n_graph.device)

    edge_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      sum_n_edge = edge_set.indices.senders.shape[0]
      edge_gr_idx = graph_idx.repeat_interleave(edge_set.n_edge)[:sum_n_edge]
      
      if isinstance(edge_set.features, dict):
        edge_features[edge_set_key.name] = {
            k: edge_aggregation_fn(v, edge_gr_idx, n_graph)
            for k, v in edge_set.features.items()
        }
      else:
        edge_features[edge_set_key.name] = edge_aggregation_fn(
            edge_set.features, edge_gr_idx, n_graph)

    node_features = {}
    for node_set_key, node_set in graph.nodes.items():
      if isinstance(node_set.features, dict):
        sum_n_node = next(iter(node_set.features.values())).shape[0]
      else:
        sum_n_node = node_set.features.shape[0]
      
      node_gr_idx = graph_idx.repeat_interleave(node_set.n_node)[:sum_n_node]
      
      if isinstance(node_set.features, dict):
        node_features[node_set_key] = {
            k: node_aggregation_fn(v, node_gr_idx, n_graph)
            for k, v in node_set.features.items()
        }
      else:
        node_features[node_set_key] = node_aggregation_fn(
            node_set.features, node_gr_idx, n_graph)

    new_features = global_fn(node_features, edge_features, graph.context.features)
    return graph.context._replace(features=new_features)


def GraphMapFeatures(
    embed_edge_fn: Optional[Mapping[str, Callable]] = None,
    embed_node_fn: Optional[Mapping[str, Callable]] = None,
    embed_global_fn: Optional[Callable] = None) -> Callable:
  """Returns function which embeds the components of a graph independently."""

  def _embed(graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    updated_edges = dict(graph.edges)
    if embed_edge_fn:
      for edge_set_name, embed_fn in embed_edge_fn.items():
        edge_set_key = graph.edge_key_by_name(edge_set_name)
        edge_set = graph.edges[edge_set_key]
        updated_edges[edge_set_key] = edge_set._replace(
            features=embed_fn(edge_set.features))

    updated_nodes = dict(graph.nodes)
    if embed_node_fn:
      for node_set_key, embed_fn in embed_node_fn.items():
        node_set = graph.nodes[node_set_key]
        updated_nodes[node_set_key] = node_set._replace(
            features=embed_fn(node_set.features))

    updated_context = graph.context
    if embed_global_fn:
      updated_context = updated_context._replace(
          features=embed_global_fn(updated_context.features))

    return graph._replace(edges=updated_edges, nodes=updated_nodes,
                          context=updated_context)

  return _embed
