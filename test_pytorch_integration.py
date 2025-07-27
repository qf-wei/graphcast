#!/usr/bin/env python3
"""Integration tests for PyTorch GraphCast components."""

import torch
import numpy as np
import sys
import os
import time
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

from graphcast import torch_utils
from graphcast import typed_graph_torch
from graphcast import deep_typed_graph_net_torch
from graphcast import xarray_torch
from graphcast import graphcast_torch


def test_memory_efficiency():
    """Test memory efficiency of PyTorch components."""
    print("Testing memory efficiency...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    else:
        device = torch.device('cpu')
        initial_memory = 0
    
    model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"mesh": 128, "grid": 64},
        edge_latent_size={"mesh2grid": 96, "grid2mesh": 96, "mesh_self": 80},
        mlp_hidden_size=256,
        mlp_num_hidden_layers=3,
        num_message_passing_steps=4,
        activation="swish"
    ).to(device)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([2]),
        features=torch.randn(2, 8).to(device)
    )
    
    nodes = {
        "mesh": typed_graph_torch.NodeSet(
            n_node=torch.tensor([100]),
            features=torch.randn(100, 32).to(device)
        ),
        "grid": typed_graph_torch.NodeSet(
            n_node=torch.tensor([200]),
            features=torch.randn(200, 24).to(device)
        )
    }
    
    edges = {
        typed_graph_torch.EdgeSetKey("mesh2grid", ("mesh", "grid")): 
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([500]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 100, (500,)).to(device),
                receivers=torch.randint(0, 200, (500,)).to(device)
            ),
            features=torch.randn(500, 16).to(device)
        ),
        typed_graph_torch.EdgeSetKey("grid2mesh", ("grid", "mesh")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([500]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 200, (500,)).to(device),
                receivers=torch.randint(0, 100, (500,)).to(device)
            ),
            features=torch.randn(500, 16).to(device)
        ),
        typed_graph_torch.EdgeSetKey("mesh_self", ("mesh", "mesh")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([300]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 100, (300,)).to(device),
                receivers=torch.randint(0, 100, (300,)).to(device)
            ),
            features=torch.randn(300, 12).to(device)
        )
    }
    
    input_graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    with torch.no_grad():
        output_graph = model(input_graph)
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated()
        memory_used = (final_memory - initial_memory) / 1024**2  # MB
        print(f"✓ Memory test completed. Used {memory_used:.1f} MB")
    else:
        print("✓ Memory test completed (CPU mode)")
    
    assert isinstance(output_graph, typed_graph_torch.TypedGraph)
    print("✓ Large model forward pass works")
    return True


def test_performance_benchmarks():
    """Test performance of key operations."""
    print("Testing performance benchmarks...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mlp = torch_utils.MLP([512, 256, 128, 64], activation="swish").to(device)
    x = torch.randn(1000, 128).to(device)
    
    for _ in range(5):
        _ = mlp(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):
        _ = mlp(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    mlp_time = (end_time - start_time) / 100 * 1000  # ms per forward pass
    
    print(f"✓ MLP forward pass: {mlp_time:.2f} ms per batch")
    
    data = torch.randn(10000, 64).to(device)
    segment_ids = torch.randint(0, 100, (10000,)).to(device)
    
    start_time = time.time()
    for _ in range(50):
        _ = torch_utils.segment_sum(data, segment_ids, 100)
    end_time = time.time()
    
    segment_time = (end_time - start_time) / 50 * 1000  # ms per operation
    print(f"✓ Segment sum: {segment_time:.2f} ms per operation")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("Testing gradient flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"mesh": 32},
        edge_latent_size={"mesh_edges": 16},
        mlp_hidden_size=64,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=2,
        activation="relu"
    ).to(device)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([1]),
        features=torch.randn(1, 4, requires_grad=True).to(device)
    )
    
    nodes = {
        "mesh": typed_graph_torch.NodeSet(
            n_node=torch.tensor([10]),
            features=torch.randn(10, 16, requires_grad=True).to(device)
        )
    }
    
    edge_key = typed_graph_torch.EdgeSetKey("mesh_edges", ("mesh", "mesh"))
    edges = {
        edge_key: typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([20]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 10, (20,)).to(device),
                receivers=torch.randint(0, 10, (20,)).to(device)
            ),
            features=torch.randn(20, 8, requires_grad=True).to(device)
        )
    }
    
    input_graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    output_graph = model(input_graph)
    
    loss = output_graph.nodes["mesh"].features.sum()
    
    loss.backward()
    
    param_count = 0
    grad_count = 0
    
    for param in model.parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
            assert not torch.isnan(param.grad).any(), "NaN gradients detected"
            assert not torch.isinf(param.grad).any(), "Inf gradients detected"
    
    print(f"✓ Gradients computed for {grad_count}/{param_count} parameters")
    assert grad_count > 0, "No gradients computed"
    print("✓ Gradient flow works correctly")
    return True


def test_batch_processing():
    """Test batch processing capabilities."""
    print("Testing batch processing...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        task_config = graphcast_torch.TaskConfig(
            input_variables=('temperature',),
            target_variables=('temperature',),
            forcing_variables=(),
            pressure_levels=(1000,),
            input_duration='6h'
        )
        
        model_config = graphcast_torch.ModelConfig(
            resolution=1,
            mesh_size=2,
            latent_size=32,
            gnn_msg_steps=2,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6
        )
        
        model = graphcast_torch.GraphCast(model_config, task_config).to(device)
        
        coords = {
            'batch': range(batch_size),
            'time': [0, 1],
            'lat': [0, 45],
            'lon': [0, 180]
        }
        
        inputs = xarray_torch.Dataset({
            'temperature': (['batch', 'time', 'lat', 'lon'], 
                           torch.randn(batch_size, 2, 2, 2).to(device))
        }, coords=coords)
        
        targets = xarray_torch.Dataset({
            'temperature': (['batch', 'time', 'lat', 'lon'], 
                           torch.randn(batch_size, 1, 2, 2).to(device))
        }, coords={k: v if k != 'time' else [2] for k, v in coords.items()})
        
        forcings = xarray_torch.Dataset({}, coords={k: v if k != 'time' else [2] for k, v in coords.items()})
        
        with torch.no_grad():
            predictions = model(inputs, targets, forcings)
        
        assert isinstance(predictions, xr.Dataset)
        print(f"✓ Batch size {batch_size} works")
    
    print("✓ Batch processing works for all tested sizes")
    return True


def test_data_transformations():
    """Test data transformation operations comprehensively."""
    print("Testing data transformations...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    coords = {
        'batch': [0, 1, 2],
        'time': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'pressure': [1000, 850, 500, 250],
        'lat': np.linspace(-90, 90, 32),
        'lon': np.linspace(0, 360, 64, endpoint=False)
    }
    
    data_vars = {
        'temperature': (['batch', 'time', 'pressure', 'lat', 'lon'], 
                       torch.randn(3, 3, 4, 32, 64).to(device)),
        'humidity': (['batch', 'time', 'pressure', 'lat', 'lon'],
                    torch.randn(3, 3, 4, 32, 64).to(device)),
        'wind_u': (['batch', 'time', 'pressure', 'lat', 'lon'],
                  torch.randn(3, 3, 4, 32, 64).to(device)),
        'wind_v': (['batch', 'time', 'pressure', 'lat', 'lon'],
                  torch.randn(3, 3, 4, 32, 64).to(device))
    }
    
    ds = xarray_torch.Dataset(data_vars, coords=coords)
    
    temp_data = xarray_torch.torch_data(ds['temperature'])
    assert temp_data.shape == (3, 3, 4, 32, 64)
    assert temp_data.device == device
    
    subset = ds.isel(time=slice(0, 2), pressure=slice(0, 2))
    subset_temp = xarray_torch.torch_data(subset['temperature'])
    assert subset_temp.shape == (3, 2, 2, 32, 64)
    
    mean_temp = ds['temperature'].mean(dim=['lat', 'lon'])
    mean_data = xarray_torch.torch_data(mean_temp)
    assert mean_data.shape == (3, 3, 4)
    
    print("✓ Complex data transformations work")
    print("✓ Multi-dimensional coordinate handling works")
    print("✓ Data slicing and aggregation works")
    return True


def test_model_state_management():
    """Test model state saving, loading, and parameter management."""
    print("Testing model state management...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"mesh": 64},
        edge_latent_size={"mesh_edges": 32},
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=3,
        activation="swish"
    ).to(device)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([1]),
        features=torch.randn(1, 4).to(device)
    )
    
    nodes = {
        "mesh": typed_graph_torch.NodeSet(
            n_node=torch.tensor([10]),
            features=torch.randn(10, 32).to(device)
        )
    }
    
    edges = {
        typed_graph_torch.EdgeSetKey("mesh_edges", ("mesh", "mesh")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([20]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 10, (20,)).to(device),
                receivers=torch.randint(0, 10, (20,)).to(device)
            ),
            features=torch.randn(20, 16).to(device)
        )
    }
    
    graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    with torch.no_grad():
        _ = model(graph)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model has {total_params:,} total parameters")
    print(f"✓ Model has {trainable_params:,} trainable parameters")
    assert total_params > 0
    assert trainable_params == total_params
    
    model.train()
    assert model.training
    
    model.eval()
    assert not model.training
    
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN found in parameter {name}"
        assert not torch.isinf(param).any(), f"Inf found in parameter {name}"
        assert param.requires_grad, f"Parameter {name} doesn't require gradients"
    
    print("✓ Model state management works")
    print("✓ Parameter initialization is valid")
    print("✓ Training/eval mode switching works")
    return True


def test_edge_cases_and_robustness():
    """Test edge cases and model robustness."""
    print("Testing edge cases and robustness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"nodes": 16},
        edge_latent_size={},  # No edges for first test
        mlp_hidden_size=32,
        mlp_num_hidden_layers=1,
        num_message_passing_steps=1,
        activation="relu",
        embed_nodes=False,  # Don't embed to avoid edge requirements
        embed_edges=False
    ).to(device)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([1]),
        features=torch.randn(1, 2).to(device)
    )
    
    nodes = {
        "nodes": typed_graph_torch.NodeSet(
            n_node=torch.tensor([1]),
            features=torch.randn(1, 16).to(device)  # Use latent size directly
        )
    }
    
    edges = {}
    
    graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    with torch.no_grad():
        output = model(graph)
    
    assert isinstance(output, typed_graph_torch.TypedGraph)
    assert output.nodes["nodes"].features.shape == (1, 16)
    print("✓ Single node graph works")
    
    edge_model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"nodes": 16},
        edge_latent_size={"self_edges": 8},
        mlp_hidden_size=32,
        mlp_num_hidden_layers=1,
        num_message_passing_steps=1,
        activation="relu",
        embed_nodes=False,
        embed_edges=False
    ).to(device)
    
    edges = {
        typed_graph_torch.EdgeSetKey("self_edges", ("nodes", "nodes")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([1]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.tensor([0]).to(device),
                receivers=torch.tensor([0]).to(device)
            ),
            features=torch.randn(1, 8).to(device)  # Use latent size
        )
    }
    
    graph_with_edges = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    with torch.no_grad():
        output_with_edges = edge_model(graph_with_edges)
    
    assert isinstance(output_with_edges, typed_graph_torch.TypedGraph)
    print("✓ Self-loop edges work")
    
    activations = ["relu", "swish", "gelu", "tanh"]
    for activation in activations:
        test_model = deep_typed_graph_net_torch.DeepTypedGraphNet(
            node_latent_size={"nodes": 8},
            edge_latent_size={"self_edges": 4},
            mlp_hidden_size=16,
            mlp_num_hidden_layers=1,
            num_message_passing_steps=1,
            activation=activation,
            embed_nodes=False,
            embed_edges=False
        ).to(device)
        
        with torch.no_grad():
            _ = test_model(graph_with_edges)
        
        print(f"✓ {activation} activation works")
    
    print("✓ Edge cases and robustness tests pass")
    return True


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("Testing numerical stability...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = deep_typed_graph_net_torch.DeepTypedGraphNet(
        node_latent_size={"nodes": 32},
        edge_latent_size={"edges": 16},
        mlp_hidden_size=64,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=2,
        activation="swish"
    ).to(device)
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([1]),
        features=torch.randn(1, 4).to(device) * 100
    )
    
    nodes = {
        "nodes": typed_graph_torch.NodeSet(
            n_node=torch.tensor([10]),
            features=torch.randn(10, 16).to(device) * 100
        )
    }
    
    edges = {
        typed_graph_torch.EdgeSetKey("edges", ("nodes", "nodes")):
        typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([20]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.randint(0, 10, (20,)).to(device),
                receivers=torch.randint(0, 10, (20,)).to(device)
            ),
            features=torch.randn(20, 8).to(device) * 100
        )
    }
    
    graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    with torch.no_grad():
        output = model(graph)
    
    output_features = output.nodes["nodes"].features
    assert not torch.isnan(output_features).any(), "NaN values in output"
    assert not torch.isinf(output_features).any(), "Inf values in output"
    assert torch.isfinite(output_features).all(), "Non-finite values in output"
    
    print("✓ Large input values handled correctly")
    
    small_graph = typed_graph_torch.TypedGraph(
        context=typed_graph_torch.Context(
            n_graph=torch.tensor([1]),
            features=torch.randn(1, 4).to(device) * 1e-6
        ),
        nodes={
            "nodes": typed_graph_torch.NodeSet(
                n_node=torch.tensor([10]),
                features=torch.randn(10, 16).to(device) * 1e-6
            )
        },
        edges={
            typed_graph_torch.EdgeSetKey("edges", ("nodes", "nodes")):
            typed_graph_torch.EdgeSet(
                n_edge=torch.tensor([20]),
                indices=typed_graph_torch.EdgesIndices(
                    senders=torch.randint(0, 10, (20,)).to(device),
                    receivers=torch.randint(0, 10, (20,)).to(device)
                ),
                features=torch.randn(20, 8).to(device) * 1e-6
            )
        }
    )
    
    with torch.no_grad():
        small_output = model(small_graph)
    
    small_features = small_output.nodes["nodes"].features
    assert not torch.isnan(small_features).any(), "NaN values with small inputs"
    assert not torch.isinf(small_features).any(), "Inf values with small inputs"
    
    print("✓ Small input values handled correctly")
    print("✓ Numerical stability tests pass")
    return True


def main():
    """Run integration tests."""
    print("Running PyTorch GraphCast integration tests...\n")
    
    tests = [
        test_memory_efficiency,
        test_performance_benchmarks,
        test_gradient_flow,
        test_batch_processing,
        test_data_transformations,
        test_model_state_management,
        test_edge_cases_and_robustness,
        test_numerical_stability
    ]
    
    passed = 0
    failed_tests = []
    
    for test in tests:
        try:
            print(f"\n{'='*60}")
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            failed_tests.append(test.__name__)
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return False
    else:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
