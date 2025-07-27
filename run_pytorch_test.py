#!/usr/bin/env python3
"""Simple script to test PyTorch conversion components."""

import sys
import os
import torch

sys.path.insert(0, '/home/ubuntu/repos/graphcast')

def test_basic_imports():
    """Test that we can import the PyTorch modules."""
    print("Testing basic imports...")
    
    try:
        from graphcast import torch_utils
        print("✓ torch_utils imported")
        
        from graphcast import typed_graph_torch
        print("✓ typed_graph_torch imported")
        
        from graphcast import mlp
        print("✓ mlp imported")
        
        from graphcast import deep_typed_graph_net_torch
        print("✓ deep_typed_graph_net_torch imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_torch_utils():
    """Test torch utilities."""
    print("\nTesting torch_utils...")
    
    try:
        from graphcast import torch_utils
        
        x = torch.randn(5, 3)
        relu_fn = torch_utils.get_activation_fn("relu")
        swish_fn = torch_utils.get_activation_fn("swish")
        
        relu_out = relu_fn(x)
        swish_out = swish_fn(x)
        
        assert relu_out.shape == x.shape
        assert swish_out.shape == x.shape
        print("✓ Activation functions work")
        
        mlp_model = torch_utils.MLP([32, 16, 8], activation="relu")
        out = mlp_model(x)
        assert out.shape == (5, 8)
        print("✓ MLP works")
        
        return True
    except Exception as e:
        print(f"✗ torch_utils test failed: {e}")
        return False

def test_typed_graph():
    """Test TypedGraph structures."""
    print("\nTesting TypedGraph...")
    
    try:
        from graphcast import typed_graph_torch
        
        context = typed_graph_torch.Context(
            n_graph=torch.tensor([1]),
            features=torch.randn(1, 4)
        )
        
        nodes = {
            "mesh": typed_graph_torch.NodeSet(
                n_node=torch.tensor([5]),
                features=torch.randn(5, 3)
            )
        }
        
        edge_key = typed_graph_torch.EdgeSetKey("mesh_edges", ("mesh", "mesh"))
        edges = {
            edge_key: typed_graph_torch.EdgeSet(
                n_edge=torch.tensor([6]),
                indices=typed_graph_torch.EdgesIndices(
                    senders=torch.tensor([0, 1, 2, 3, 4, 0]),
                    receivers=torch.tensor([1, 2, 3, 4, 0, 1])
                ),
                features=torch.randn(6, 2)
            )
        }
        
        graph = typed_graph_torch.TypedGraph(
            context=context,
            nodes=nodes,
            edges=edges
        )
        
        edge_set = graph.edge_by_name("mesh_edges")
        assert edge_set.features.shape == (6, 2)
        print("✓ TypedGraph works")
        
        return True
    except Exception as e:
        print(f"✗ TypedGraph test failed: {e}")
        return False

def test_deep_typed_graph_net():
    """Test DeepTypedGraphNet."""
    print("\nTesting DeepTypedGraphNet...")
    
    try:
        from graphcast import deep_typed_graph_net_torch
        
        model = deep_typed_graph_net_torch.DeepTypedGraphNet(
            node_latent_size={"mesh": 32},
            edge_latent_size={"mesh_edges": 16},
            mlp_hidden_size=64,
            mlp_num_hidden_layers=2,
            num_message_passing_steps=2,
            activation="relu"
        )
        
        assert hasattr(model, 'forward')
        assert hasattr(model, '_node_latent_size')
        print("✓ DeepTypedGraphNet instantiation works")
        
        return True
    except Exception as e:
        print(f"✗ DeepTypedGraphNet test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running PyTorch conversion tests...\n")
    
    tests = [
        test_basic_imports,
        test_torch_utils,
        test_typed_graph,
        test_deep_typed_graph_net
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
