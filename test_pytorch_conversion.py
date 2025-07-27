#!/usr/bin/env python3
"""Comprehensive test script to verify PyTorch conversion components work correctly."""

import torch
import numpy as np
import sys
import os
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

from graphcast import torch_utils
from graphcast import typed_graph_torch
from graphcast import mlp
from graphcast import deep_typed_graph_net_torch
from graphcast import xarray_torch
from graphcast import graphcast_torch


def test_torch_utils():
    """Test torch utilities comprehensively."""
    print("Testing torch_utils...")
    
    x = torch.randn(10, 5)
    activations = ["relu", "swish", "silu", "gelu", "tanh", "sigmoid"]
    
    for act_name in activations:
        try:
            act_fn = torch_utils.get_activation_fn(act_name)
            out = act_fn(x)
            assert out.shape == x.shape
            assert torch.is_tensor(out)
        except ValueError:
            continue  # Skip unsupported activations
    print("✓ Activation functions work")
    
    mlp_configs = [
        ([64, 32, 16], "relu", False),
        ([32, 64, 32], "swish", True),
        ([16, 8], "gelu", False)
    ]
    
    for output_sizes, activation, activate_final in mlp_configs:
        mlp_model = torch_utils.MLP(
            output_sizes=output_sizes,
            activation=activation,
            activate_final=activate_final,
            use_layer_norm=True
        )
        out = mlp_model(x)
        assert out.shape == (10, output_sizes[-1])
        assert torch.is_tensor(out)
    print("✓ MLP configurations work")
    
    data = torch.randn(20, 8)
    segment_ids = torch.randint(0, 5, (20,))
    num_segments = 5
    
    sum_result = torch_utils.segment_sum(data, segment_ids, num_segments)
    mean_result = torch_utils.segment_mean(data, segment_ids, num_segments)
    
    assert sum_result.shape == (5, 8)
    assert mean_result.shape == (5, 8)
    assert torch.is_tensor(sum_result)
    assert torch.is_tensor(mean_result)
    print("✓ Segment operations work")
    
    mlp_with_norm = torch_utils.make_mlp_with_norm_conditioning(
        output_sizes=[32, 16],
        use_norm_conditioning=True
    )
    norm_conditioning = torch.randn(10, 12)
    out = mlp_with_norm(x, norm_conditioning)
    assert out.shape == (10, 16)
    print("✓ MLP with norm conditioning works")
    return True


def test_mlp_norm_conditioning():
    """Test LinearNormConditioning module comprehensively."""
    print("Testing LinearNormConditioning...")
    
    norm_cond = mlp.LinearNormConditioning()
    
    test_cases = [
        (torch.randn(5, 10), torch.randn(5, 8)),
        (torch.randn(3, 7, 15), torch.randn(3, 7, 12)),
        (torch.randn(1, 20), torch.randn(1, 16))
    ]
    
    for inputs, conditioning in test_cases:
        output = norm_cond(inputs, conditioning)
        assert output.shape == inputs.shape
        assert torch.is_tensor(output)
        assert not torch.allclose(output, inputs, atol=1e-3)
    
    print("✓ LinearNormConditioning works with various shapes")
    return True


def test_typed_graph():
    """Test TypedGraph data structures comprehensively."""
    print("Testing TypedGraph...")
    
    context = typed_graph_torch.Context(
        n_graph=torch.tensor([2]),  # Test with batch size > 1
        features=torch.randn(2, 4)
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
    
    edge_key1 = typed_graph_torch.EdgeSetKey("mesh2grid", ("mesh", "grid"))
    edge_key2 = typed_graph_torch.EdgeSetKey("grid2mesh", ("grid", "mesh"))
    edge_key3 = typed_graph_torch.EdgeSetKey("mesh_self", ("mesh", "mesh"))
    
    edges = {
        edge_key1: typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([6]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.tensor([0, 1, 2, 3, 4, 0]),
                receivers=torch.tensor([0, 1, 2, 3, 0, 1])
            ),
            features=torch.randn(6, 2)
        ),
        edge_key2: typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([4]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.tensor([0, 1, 2, 3]),
                receivers=torch.tensor([1, 2, 3, 4])
            ),
            features=torch.randn(4, 3)
        ),
        edge_key3: typed_graph_torch.EdgeSet(
            n_edge=torch.tensor([8]),
            indices=typed_graph_torch.EdgesIndices(
                senders=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
                receivers=torch.tensor([1, 2, 3, 4, 0, 2, 3, 4])
            ),
            features=torch.randn(8, 2)
        )
    }
    
    graph = typed_graph_torch.TypedGraph(
        context=context,
        nodes=nodes,
        edges=edges
    )
    
    edge_set1 = graph.edge_by_name("mesh2grid")
    edge_set2 = graph.edge_by_name("grid2mesh")
    edge_set3 = graph.edge_by_name("mesh_self")
    
    assert edge_set1.features.shape == (6, 2)
    assert edge_set2.features.shape == (4, 3)
    assert edge_set3.features.shape == (8, 2)
    print("✓ Multiple edge types work")
    
    if torch.cuda.is_available():
        graph_cuda = graph.to_device(torch.device('cuda'))
        assert graph_cuda.context.features.device.type == 'cuda'
        assert graph_cuda.nodes["mesh"].features.device.type == 'cuda'
        assert graph_cuda.edges[edge_key1].features.device.type == 'cuda'
        print("✓ Device movement works")
    
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 3
    assert graph.context.n_graph.item() == 2
    print("✓ TypedGraph comprehensive tests work")
    return True


def test_deep_typed_graph_net():
    """Test DeepTypedGraphNet instantiation and forward pass."""
    print("Testing DeepTypedGraphNet...")
    
    try:
        configs = [
            {
                "node_latent_size": {"mesh": 64, "grid": 32},
                "edge_latent_size": {"mesh2grid": 48, "grid2mesh": 48},
                "mlp_hidden_size": 128,
                "mlp_num_hidden_layers": 2,
                "num_message_passing_steps": 3,
                "activation": "relu"
            },
            {
                "node_latent_size": {"mesh": 32},
                "edge_latent_size": {"mesh_self": 24},
                "mlp_hidden_size": 64,
                "mlp_num_hidden_layers": 1,
                "num_message_passing_steps": 2,
                "activation": "swish",
                "use_layer_norm": True,
                "use_norm_conditioning": True
            }
        ]
        
        for i, config in enumerate(configs):
            model = deep_typed_graph_net_torch.DeepTypedGraphNet(**config)
            assert hasattr(model, 'forward')
            assert hasattr(model, '_node_latent_size')
            print(f"✓ DeepTypedGraphNet config {i+1} instantiation works")
        
        model = deep_typed_graph_net_torch.DeepTypedGraphNet(
            node_latent_size={"mesh": 32},
            edge_latent_size={"mesh_edges": 16},
            mlp_hidden_size=64,
            mlp_num_hidden_layers=2,
            num_message_passing_steps=2,
            activation="relu"
        )
        
        context = typed_graph_torch.Context(
            n_graph=torch.tensor([1]),
            features=torch.randn(1, 4)
        )
        
        nodes = {
            "mesh": typed_graph_torch.NodeSet(
                n_node=torch.tensor([5]),
                features=torch.randn(5, 16)  # Input features
            )
        }
        
        edge_key = typed_graph_torch.EdgeSetKey("mesh_edges", ("mesh", "mesh"))
        edges = {
            edge_key: typed_graph_torch.EdgeSet(
                n_edge=torch.tensor([8]),
                indices=typed_graph_torch.EdgesIndices(
                    senders=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
                    receivers=torch.tensor([1, 2, 3, 4, 0, 2, 3, 4])
                ),
                features=torch.randn(8, 8)  # Input edge features
            )
        }
        
        input_graph = typed_graph_torch.TypedGraph(
            context=context,
            nodes=nodes,
            edges=edges
        )
        
        with torch.no_grad():
            output_graph = model(input_graph)
            
        assert isinstance(output_graph, typed_graph_torch.TypedGraph)
        assert "mesh" in output_graph.nodes
        assert output_graph.nodes["mesh"].features.shape[0] == 5  # Same number of nodes
        assert output_graph.nodes["mesh"].features.shape[1] == 32  # Latent size
        print("✓ DeepTypedGraphNet forward pass works")
        
    except Exception as e:
        print(f"✗ DeepTypedGraphNet failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_xarray_torch():
    """Test xarray-PyTorch integration."""
    print("Testing xarray_torch...")
    
    torch_tensor = torch.randn(3, 4, 5)
    wrapped = xarray_torch.wrap(torch_tensor)
    unwrapped = xarray_torch.unwrap(wrapped)
    
    assert torch.allclose(torch_tensor, unwrapped)
    print("✓ Tensor wrapping/unwrapping works")
    
    var = xarray_torch.Variable(['x', 'y'], torch.randn(3, 4))
    assert var.dims == ('x', 'y')
    assert var.shape == (3, 4)
    print("✓ Variable creation works")
    
    data = torch.randn(2, 3, 4)
    coords = {'time': [0, 1], 'lat': [10, 20, 30], 'lon': [0, 90, 180, 270]}
    da = xarray_torch.DataArray(data, dims=['time', 'lat', 'lon'], coords=coords)
    
    assert da.shape == (2, 3, 4)
    assert 'time' in da.coords
    assert 'lat' in da.coords
    assert 'lon' in da.coords
    print("✓ DataArray creation works")
    
    ds = xarray_torch.Dataset({
        'temperature': (['time', 'lat', 'lon'], torch.randn(2, 3, 4)),
        'pressure': (['time', 'lat', 'lon'], torch.randn(2, 3, 4))
    }, coords=coords)
    
    assert 'temperature' in ds.data_vars
    assert 'pressure' in ds.data_vars
    assert ds['temperature'].shape == (2, 3, 4)
    print("✓ Dataset creation works")
    
    torch_temp = xarray_torch.torch_data(ds['temperature'])
    assert torch.is_tensor(torch_temp)
    assert torch_temp.shape == (2, 3, 4)
    print("✓ torch_data extraction works")
    return True


def test_graphcast_model():
    """Test GraphCast model instantiation and basic operations."""
    print("Testing GraphCast model...")
    
    try:
        task_config = graphcast_torch.TaskConfig(
            input_variables=('2m_temperature', 'mean_sea_level_pressure'),
            target_variables=('2m_temperature', 'mean_sea_level_pressure'),
            forcing_variables=('toa_incident_solar_radiation',),
            pressure_levels=(1000, 850, 500, 250),
            input_duration='12h'
        )
        
        model_config = graphcast_torch.ModelConfig(
            resolution=1,
            mesh_size=4,
            latent_size=64,
            gnn_msg_steps=6,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6
        )
        
        model = graphcast_torch.GraphCast(model_config, task_config)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'loss')
        print("✓ GraphCast model instantiation works")
        
        batch_size = 1
        time_steps = 2
        lat_points = 4
        lon_points = 8
        
        coords = {
            'batch': range(batch_size),
            'time': range(time_steps),
            'lat': np.linspace(-90, 90, lat_points),
            'lon': np.linspace(0, 360, lon_points, endpoint=False)
        }
        
        inputs = xarray_torch.Dataset({
            '2m_temperature': (['batch', 'time', 'lat', 'lon'], 
                              torch.randn(batch_size, time_steps, lat_points, lon_points)),
            'mean_sea_level_pressure': (['batch', 'time', 'lat', 'lon'],
                                       torch.randn(batch_size, time_steps, lat_points, lon_points))
        }, coords=coords)
        
        targets = xarray_torch.Dataset({
            '2m_temperature': (['batch', 'time', 'lat', 'lon'], 
                              torch.randn(batch_size, 1, lat_points, lon_points)),
            'mean_sea_level_pressure': (['batch', 'time', 'lat', 'lon'],
                                       torch.randn(batch_size, 1, lat_points, lon_points))
        }, coords={k: v if k != 'time' else [time_steps] for k, v in coords.items()})
        
        forcings = xarray_torch.Dataset({
            'toa_incident_solar_radiation': (['batch', 'time', 'lat', 'lon'],
                                           torch.randn(batch_size, 1, lat_points, lon_points))
        }, coords={k: v if k != 'time' else [time_steps] for k, v in coords.items()})
        
        with torch.no_grad():
            predictions = model(inputs, targets, forcings)
            
        assert isinstance(predictions, xr.Dataset)
        assert '2m_temperature' in predictions.data_vars
        assert 'mean_sea_level_pressure' in predictions.data_vars
        print("✓ GraphCast forward pass works")
        
        loss, diagnostics = model.loss(inputs, targets, forcings)
        assert torch.is_tensor(loss)
        assert loss.numel() == 1  # Scalar loss
        assert isinstance(diagnostics, xr.Dataset)
        print("✓ GraphCast loss computation works")
        
    except Exception as e:
        print(f"✗ GraphCast model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("Testing device compatibility...")
    
    x = torch.randn(5, 10)
    mlp = torch_utils.MLP([32, 16], activation="relu")
    out_cpu = mlp(x)
    assert out_cpu.device.type == 'cpu'
    print("✓ CPU computation works")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x_gpu = x.to(device)
        mlp_gpu = mlp.to(device)
        out_gpu = mlp_gpu(x_gpu)
        assert out_gpu.device.type == 'cuda'
        print("✓ GPU computation works")
    else:
        print("✓ GPU not available, skipping GPU tests")
    return True


def main():
    """Run all comprehensive tests."""
    print("Running comprehensive PyTorch conversion tests...\n")
    
    tests = [
        test_torch_utils,
        test_mlp_norm_conditioning,
        test_typed_graph,
        test_deep_typed_graph_net,
        test_xarray_torch,
        test_graphcast_model,
        test_device_compatibility
    ]
    
    passed = 0
    failed_tests = []
    
    for test in tests:
        try:
            print(f"\n{'='*50}")
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed_tests.append(test.__name__)
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed_tests.append(test.__name__)
            print(f"❌ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return False
    else:
        print("🎉 ALL TESTS PASSED!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
