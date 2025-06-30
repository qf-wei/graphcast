#!/usr/bin/env python3
"""Test JAX CUDA installation and GPU detection."""

import jax
import jax.numpy as jnp

def test_gpu():
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"CUDA available: {any('gpu' in str(d) for d in jax.devices())}")
    
    if jax.devices() and 'gpu' in str(jax.devices()[0]):
        print("GPU detected! Testing basic computation...")
        x = jnp.array([1, 2, 3, 4, 5])
        y = x * 2
        print(f"Test computation: {x} * 2 = {y}")
        print("GPU test successful!")
        return True
    else:
        print("No GPU detected or CUDA not available")
        return False

if __name__ == "__main__":
    test_gpu()
