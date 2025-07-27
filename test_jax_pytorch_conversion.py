#!/usr/bin/env python3
"""Test JAX to PyTorch parameter conversion."""

import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from graphcast import graphcast_torch
from convert_jax_to_pytorch import (
    convert_jax_array_to_torch,
    map_jax_params_to_pytorch,
    create_pytorch_model_from_jax_config,
    convert_jax_checkpoint_to_pytorch
)

def create_mock_jax_checkpoint():
    """Create a mock JAX checkpoint for testing."""
    
    model_config = graphcast_torch.ModelConfig(
        resolution=0,
        mesh_size=6,
        latent_size=64,
        gnn_msg_steps=2,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=('geopotential', 'temperature'),
        target_variables=('geopotential', 'temperature'),
        forcing_variables=('mean_sea_level_pressure',),
        pressure_levels=(500, 700, 850),
        input_duration='12h'
    )
    
    mock_jax_params = {
        'grid2mesh_gnn': {
            'processor_networks': {
                '0': {
                    'update_edge_fn': {
                        'grid2mesh': {
                            'mlp': {
                                'layers': {
                                    '0': {
                                        'w': np.random.randn(64, 256).astype(np.float32),
                                        'b': np.random.randn(64).astype(np.float32)
                                    },
                                    '1': {
                                        'w': np.random.randn(64, 64).astype(np.float32),
                                        'b': np.random.randn(64).astype(np.float32)
                                    }
                                }
                            }
                        }
                    },
                    'update_node_fn': {
                        'mesh_nodes': {
                            'mlp': {
                                'layers': {
                                    '0': {
                                        'w': np.random.randn(64, 384).astype(np.float32),
                                        'b': np.random.randn(64).astype(np.float32)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    mock_model_config = {
        'resolution': 0,
        'mesh_size': 6,
        'latent_size': 64,
        'gnn_msg_steps': 2,
        'hidden_layers': 1,
        'radius_query_fraction_edge_length': 0.6
    }
    
    mock_task_config = {
        'input_variables': ('geopotential', 'temperature'),
        'target_variables': ('geopotential', 'temperature'),
        'forcing_variables': ('mean_sea_level_pressure',),
        'pressure_levels': (500, 700, 850),
        'input_duration': '12h'
    }
    
    return {
        'params': mock_jax_params,
        'model_config': mock_model_config,
        'task_config': mock_task_config
    }

def test_jax_array_conversion():
    """Test conversion of JAX arrays to PyTorch tensors."""
    print("Testing JAX array to PyTorch tensor conversion...")
    
    numpy_array = np.random.randn(10, 20).astype(np.float32)
    torch_tensor = convert_jax_array_to_torch(numpy_array)
    
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch_tensor.shape == (10, 20)
    assert torch_tensor.dtype == torch.float32
    assert np.allclose(torch_tensor.numpy(), numpy_array)
    
    print("✓ JAX array conversion works correctly")

def test_parameter_mapping():
    """Test mapping of JAX parameters to PyTorch format."""
    print("Testing parameter mapping...")
    
    model_config = graphcast_torch.ModelConfig(
        resolution=0,
        mesh_size=6,
        latent_size=512,
        gnn_msg_steps=16,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=('geopotential', 'temperature'),
        target_variables=('geopotential', 'temperature'),
        forcing_variables=('mean_sea_level_pressure',),
        pressure_levels=(500, 700, 850),
        input_duration='12h'
    )
    
    pytorch_model = graphcast_torch.GraphCast(model_config, task_config)
    
    mock_jax_data = create_mock_jax_checkpoint()
    jax_params = mock_jax_data['params']
    
    pytorch_state_dict = map_jax_params_to_pytorch(jax_params, pytorch_model)
    
    assert isinstance(pytorch_state_dict, dict)
    assert len(pytorch_state_dict) > 0
    
    for key, value in pytorch_state_dict.items():
        assert isinstance(value, torch.Tensor), f"Parameter {key} is not a tensor"
    
    print(f"✓ Parameter mapping created {len(pytorch_state_dict)} parameters")

def test_full_conversion():
    """Test full JAX checkpoint to PyTorch conversion."""
    print("Testing full checkpoint conversion...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        jax_checkpoint_path = os.path.join(temp_dir, 'mock_jax_checkpoint.npz')
        pytorch_checkpoint_path = os.path.join(temp_dir, 'converted_pytorch.pth')
        
        mock_data = create_mock_jax_checkpoint()
        np.savez(jax_checkpoint_path, **mock_data)
        
        pytorch_model = create_pytorch_model_from_jax_config(jax_checkpoint_path)
        
        output_path = convert_jax_checkpoint_to_pytorch(
            jax_checkpoint_path,
            pytorch_model,
            pytorch_checkpoint_path
        )
        
        assert os.path.exists(output_path)
        
        checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
        
        assert 'model_state_dict' in checkpoint
        assert 'jax_params' in checkpoint
        assert 'conversion_info' in checkpoint
        
        new_model = create_pytorch_model_from_jax_config(jax_checkpoint_path)
        new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print("✓ Full conversion pipeline works correctly")

def test_model_creation_from_config():
    """Test creating PyTorch model from JAX config."""
    print("Testing model creation from JAX config...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        jax_checkpoint_path = os.path.join(temp_dir, 'config_test.npz')
        mock_data = create_mock_jax_checkpoint()
        np.savez(jax_checkpoint_path, **mock_data)
        
        pytorch_model = create_pytorch_model_from_jax_config(jax_checkpoint_path)
        
        assert isinstance(pytorch_model, graphcast_torch.GraphCast)
        assert pytorch_model.model_config.latent_size == 64
        assert pytorch_model.task_config.input_variables == ('geopotential', 'temperature')
        
        batch_size = 2
        coords = {
            'batch': np.arange(batch_size),
            'time': np.arange(2),
            'level': np.array([500, 700, 850]),
            'lat': np.linspace(-90, 90, 8),
            'lon': np.linspace(0, 360, 16, endpoint=False)
        }
        
        inputs = {
            'geopotential': torch.randn(batch_size, 2, 3, 8, 16),
            'temperature': torch.randn(batch_size, 2, 3, 8, 16)
        }
        
        print("✓ Model creation from JAX config works correctly")

def main():
    """Run all conversion tests."""
    print("Running JAX to PyTorch conversion tests...\n")
    
    tests = [
        test_jax_array_conversion,
        test_parameter_mapping,
        test_model_creation_from_config,
        test_full_conversion
    ]
    
    passed = 0
    for test in tests:
        try:
            print(f"\n{'='*50}")
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"CONVERSION TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL CONVERSION TESTS PASSED!")
        return True
    else:
        print("❌ Some conversion tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
