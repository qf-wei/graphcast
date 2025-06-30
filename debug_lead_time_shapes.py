#!/usr/bin/env python3
"""
Debug script to understand how lead times affect grid dimensions in GraphCast
"""
import sys
sys.path.append('/home/ubuntu/graphcast')
import xarray
import numpy as np
import dataclasses
from graphcast import data_utils
from precipitation_validation import setup_gcs_client, find_2019_datasets, load_dataset, load_model_checkpoint

def debug_lead_time_shapes():
    """Debug how different lead times affect grid dimensions"""
    print("🔍 Debugging lead time effects on grid dimensions...")
    
    gcs_bucket = setup_gcs_client()
    
    params, state, task_config, sampler_config, noise_config, noise_encoder_config, denoiser_architecture_config = load_model_checkpoint(gcs_bucket)
    
    dataset_files = find_2019_datasets(gcs_bucket, "2019-03")
    dataset_1deg = [f for f in dataset_files if "res-1.0" in f]
    dataset_025deg = [f for f in dataset_files if "res-0.25" in f]
    
    if dataset_1deg:
        dataset_file = dataset_1deg[0]
        print(f"Using 1.0deg dataset: {dataset_file}")
    elif dataset_025deg:
        dataset_file = dataset_025deg[0]
        print(f"Using 0.25deg dataset: {dataset_file}")
    else:
        dataset_file = dataset_files[0]
        print(f"Using first available dataset: {dataset_file}")
    
    dataset = load_dataset(gcs_bucket, dataset_file)
    print(f"Original dataset shape: {dataset.dims}")
    print(f"Original lat shape: {dataset.lat.shape}, lon shape: {dataset.lon.shape}")
    print(f"Original grid nodes: {dataset.lat.shape[0] * dataset.lon.shape[0]}")
    
    forecast_data = dataset.isel(time=slice(0, min(10, dataset.dims["time"])))
    print(f"Forecast data shape: {forecast_data.dims}")
    
    lead_times_to_test = [12, 24, 48, 120, 240]
    
    for max_lead_time in lead_times_to_test:
        print(f"\n--- Testing lead time {max_lead_time}h ---")
        try:
            eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
                forecast_data, 
                target_lead_times=slice("12h", f"{max_lead_time}h"),
                **dataclasses.asdict(task_config)
            )
            
            print(f"Inputs shape: {eval_inputs.dims}")
            print(f"Targets shape: {eval_targets.dims}")
            print(f"Forcings shape: {eval_forcings.dims}")
            
            print(f"Inputs lat: {eval_inputs.lat.shape}, lon: {eval_inputs.lon.shape}")
            print(f"Targets lat: {eval_targets.lat.shape}, lon: {eval_targets.lon.shape}")
            
            input_grid_nodes = eval_inputs.lat.shape[0] * eval_inputs.lon.shape[0]
            target_grid_nodes = eval_targets.lat.shape[0] * eval_targets.lon.shape[0]
            
            print(f"Input grid nodes: {input_grid_nodes}")
            print(f"Target grid nodes: {target_grid_nodes}")
            
            if input_grid_nodes != target_grid_nodes:
                print(f"⚠️  MISMATCH: Input ({input_grid_nodes}) != Target ({target_grid_nodes})")
            else:
                print(f"✅ Grid nodes consistent: {input_grid_nodes}")
                
        except Exception as e:
            print(f"❌ Error with lead time {max_lead_time}h: {e}")

if __name__ == "__main__":
    debug_lead_time_shapes()
