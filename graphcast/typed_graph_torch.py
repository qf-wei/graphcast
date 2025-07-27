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
"""Data-structure for storing graphs with typed edges and nodes using PyTorch."""

from typing import NamedTuple, Any, Union, Tuple, Mapping, TypeVar
import torch

ArrayLike = Union[torch.Tensor, Any]
ArrayLikeTree = Union[Any, ArrayLike]

_T = TypeVar('_T')


class NodeSet(NamedTuple):
  """Represents a set of nodes."""
  n_node: ArrayLike
  features: ArrayLikeTree


class EdgesIndices(NamedTuple):
  """Represents indices to nodes adjacent to the edges."""
  senders: ArrayLike
  receivers: ArrayLike


class EdgeSet(NamedTuple):
  """Represents a set of edges."""
  n_edge: ArrayLike
  indices: EdgesIndices
  features: ArrayLikeTree


class Context(NamedTuple):
  n_graph: ArrayLike
  features: ArrayLikeTree


class EdgeSetKey(NamedTuple):
  name: str
  node_sets: Tuple[str, str]


class TypedGraph(NamedTuple):
  """A graph with typed nodes and edges using PyTorch tensors.

  A typed graph is made of a context, multiple sets of nodes and multiple
  sets of edges connecting those nodes (as indicated by the EdgeSetKey).
  """

  context: Context
  nodes: Mapping[str, NodeSet]
  edges: Mapping[EdgeSetKey, EdgeSet]

  def edge_key_by_name(self, name: str) -> EdgeSetKey:
    found_key = [k for k in self.edges.keys() if k.name == name]
    if len(found_key) != 1:
      raise KeyError("invalid edge key '{}'. Available edges: [{}]".format(
          name, ', '.join(x.name for x in self.edges.keys())))
    return found_key[0]

  def edge_by_name(self, name: str) -> EdgeSet:
    return self.edges[self.edge_key_by_name(name)]

  def to_device(self, device: torch.device) -> 'TypedGraph':
    """Move all tensors in the graph to the specified device."""
    def move_to_device(obj):
      if isinstance(obj, torch.Tensor):
        return obj.to(device)
      elif isinstance(obj, dict):
        return {k: move_to_device(v) for k, v in obj.items()}
      elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item) for item in obj)
      else:
        return obj

    new_context = Context(
        n_graph=move_to_device(self.context.n_graph),
        features=move_to_device(self.context.features)
    )
    
    new_nodes = {}
    for name, node_set in self.nodes.items():
      new_nodes[name] = NodeSet(
          n_node=move_to_device(node_set.n_node),
          features=move_to_device(node_set.features)
      )
    
    new_edges = {}
    for edge_key, edge_set in self.edges.items():
      new_edges[edge_key] = EdgeSet(
          n_edge=move_to_device(edge_set.n_edge),
          indices=EdgesIndices(
              senders=move_to_device(edge_set.indices.senders),
              receivers=move_to_device(edge_set.indices.receivers)
          ),
          features=move_to_device(edge_set.features)
      )
    
    return TypedGraph(
        context=new_context,
        nodes=new_nodes,
        edges=new_edges
    )
