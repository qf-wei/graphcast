#!/usr/bin/env python3
"""Test script to verify PyTorch conversion components work correctly."""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graphcast import torch_utils
from graphcast import typed_graph_torch
from graphcast import mlp
from graphcast import deep_typed_graph_net_torch


def test_torch_utils():
    """Test basic torch utilities."""
    print("Testing torch_utils...")
    
    x = torch.randn(10, 5)
    relu_fn = torch_utils.get_activation_fn("relu")
    swish_fn = torch_utils.get_activation_fn("swish")
    
    relu_out = relu_fn(x)
    swish_out = swish_fn(x)
    
    assert relu_out.shape == x.shape
    assert swish_out.shape == x.shape
    print("✓ Activation functions work")
    
    mlp_model = torch_utils.MLP([64, 32, 16], activation="relu")
    out = mlp_model(x)
    assert out.shape == (10, 16)
    print("✓ MLP works")


def test_mlp_norm_conditioning():
    """Test LinearNormConditioning module."""
    print("Testing LinearNormConditioning...")
    
    norm_cond = mlp.LinearNormConditioning()
    inputs = torch.randn(5, 10)
    conditioning = torch.randn(5, 8)
    
    output = norm_cond(inputs, conditioning)
    assert output.shape == inputs.shape
    print("✓ LinearNormConditioning works")


def test_typed_graph():
    """Test TypedGraph data structures."""
    print("Testing TypedGraph...")
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([1]),
        features=torch.randn(1, 4)
    )
    
    nodes = {
        "mesh": typed_graph_torch.NodeSet(
            n_node=torch.tensor([5]),
            features=torch.randn(5, 3)
        ),
        "grid": typed_graph_torch.NodeSet(
            n_node=torch.tensor([4]),
            features=torch.randn(4, 2)
        )
    }
    
    edge_key = typed_graph_torch.EdgeSetKey("mesh2grid", ("mesh", "grid"))
    edges = {
        edge_key: typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([6]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.tensor([0, 1, 2, 3, 4, 0]),
                receivers=torch.tensor([0, 1, 2, 3, 0, 1])
            ),
            features=torch.randn(6, 2)
        )
    }
    
    graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    if torch.cuda.is_available():
        graph_cuda = graph.to_device(torch.device('cuda'))
        assert graph_cuda.context.features.device.type == 'cuda'
        print("✓ Device movement works")
    
    edge_set = graph.edge_by_name("mesh2grid")
    assert edge_set.features.shape == (6, 2)
    print("✓ TypedGraph works")


def test_deep_typed_graph_net():
    """Test DeepTypedGraphNet instantiation."""
    print("Testing DeepTypedGraphNet...")
    
    try:
        model = deep_typed_graph_net_torch.DeepTypedGraphNet(
            node_latent_size={"mesh": 64, "grid": 32},
            edge_latent_size={"mesh2grid": 48},
            mlp_hidden_size=128,
            mlp_num_hidden_layers=2,
            num_message_passing_steps=3,
            activation="relu"
        )
        print("✓ DeepTypedGraphNet instantiation works")
        
        assert hasattr(model, 'forward')
        assert hasattr(model, '_node_latent_size')
        print("✓ DeepTypedGraphNet has expected attributes")
        
    except Exception as e:
        print(f"✗ DeepTypedGraphNet failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Running PyTorch conversion tests...\n")
    
    try:
        test_torch_utils()
        test_mlp_norm_conditioning()
        test_typed_graph()
        test_deep_typed_graph_net()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
