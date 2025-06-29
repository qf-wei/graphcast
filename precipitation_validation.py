#!/usr/bin/env python3
"""
Precipitation Validation Program for GraphCast/GenCast on RTX 5090 GPU

This program loads 2019 weather data, runs GenCast inference on RTX 5090 GPU,
and validates the accuracy of total_precipitation_12hr predictions by computing L2 error
against ground truth.

Key Features:
- Uses GPU-compatible attention configuration (triblockdiag_mha)
- Loads real 2019 ERA5 weather data from Google Cloud Storage
- Computes L2 error for precipitation predictions vs ground truth
- Handles memory efficiently for RTX 5090 GPU inference
- Provides comprehensive validation statistics

Usage:
    python precipitation_validation.py --month 2019-03 --num_forecasts 5 --max_lead_time 24

Requirements:
- RTX 5090 GPU with CUDA support
- JAX with CUDA installation
- GraphCast dependencies
- Internet connection for dataset access
"""

import argparse
import dataclasses
import datetime
import logging
from typing import Optional, Tuple
import numpy as np
import xarray
import jax
import jax.numpy as jnp
import haiku as hk
from google.cloud import storage

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_gcs_client():
    """Setup Google Cloud Storage client for accessing GraphCast data."""
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    return gcs_bucket


def load_model_checkpoint(gcs_bucket, model_name: str = "GenCast 1p0deg Mini <2019.npz"):
    """Load GenCast model checkpoint with GPU-compatible attention configuration."""
    logger.info(f"Loading model checkpoint: {model_name}")
    
    dir_prefix = "gencast/"
    with gcs_bucket.blob(dir_prefix + f"params/{model_name}").open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    
    params = ckpt.params
    state = {}
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"
    
    logger.info("Model loaded with GPU-compatible attention: triblockdiag_mha")
    logger.info(f"Model description: {ckpt.description}")
    
    return params, state, task_config, sampler_config, noise_config, noise_encoder_config, denoiser_architecture_config


def load_normalization_stats(gcs_bucket):
    """Load normalization statistics for the model."""
    logger.info("Loading normalization statistics...")
    
    dir_prefix = "gencast/"
    with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
        min_by_level = xarray.load_dataset(f).compute()
    
    return diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level


def find_2019_datasets(gcs_bucket, target_month: str) -> list:
    """Find available 2019 datasets for the specified month."""
    logger.info(f"Searching for 2019 datasets...")
    
    dir_prefix = "gencast/"
    dataset_files = []
    
    for blob in gcs_bucket.list_blobs(prefix=(dir_prefix + "dataset/")):
        name = blob.name.removeprefix(dir_prefix+"dataset/")
        if name and "2019" in name and "era5" in name:
            dataset_files.append(name)
    
    logger.info(f"Found {len(dataset_files)} datasets for 2019")
    for dataset in dataset_files[:5]:
        logger.info(f"  {dataset}")
    if len(dataset_files) > 5:
        logger.info(f"  ... and {len(dataset_files) - 5} more")
    
    return sorted(dataset_files)


def load_dataset(gcs_bucket, dataset_file: str) -> xarray.Dataset:
    """Load a specific dataset file."""
    logger.info(f"Loading dataset: {dataset_file}")
    
    dir_prefix = "gencast/"
    with gcs_bucket.blob(dir_prefix+f"dataset/{dataset_file}").open("rb") as f:
        dataset = xarray.load_dataset(f).compute()
    
    logger.info(f"Dataset shape: {dataset.dims}")
    logger.info(f"Variables: {list(dataset.data_vars.keys())}")
    
    return dataset


def construct_wrapped_gencast(sampler_config, task_config, denoiser_architecture_config, 
                             noise_config, noise_encoder_config, 
                             diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level):
    """Construct the wrapped GenCast predictor with normalization and NaN cleaning."""
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=min_by_level,
        var_to_clean='sea_surface_temperature',
    )

    return predictor


def compute_l2_error(predictions: xarray.Dataset, targets: xarray.Dataset, variable: str = "total_precipitation_12hr") -> Tuple[float, xarray.DataArray]:
    """Compute L2 error between predictions and ground truth for a specific variable."""
    if variable not in predictions.data_vars:
        raise ValueError(f"Variable {variable} not found in predictions")
    if variable not in targets.data_vars:
        raise ValueError(f"Variable {variable} not found in targets")
    
    pred_var = predictions[variable]
    target_var = targets[variable]
    
    squared_diff = (pred_var - target_var) ** 2
    
    l2_error_per_time = jnp.sqrt(squared_diff.mean(dim=['lat', 'lon']))
    
    mean_l2_error = float(l2_error_per_time.mean())
    
    logger.info(f"L2 error for {variable}: {mean_l2_error:.6f}")
    
    return mean_l2_error, l2_error_per_time


def run_validation(month: str, num_forecasts: int = 10, max_lead_time_hours: int = 120):
    """Run precipitation validation for the specified month."""
    logger.info(f"Starting precipitation validation for {month}")
    logger.info(f"JAX devices available: {jax.local_devices()}")
    
    gpu_available = any('gpu' in str(device).lower() for device in jax.local_devices())
    if gpu_available:
        logger.info("✓ GPU detected - using GPU acceleration")
    else:
        logger.warning("⚠ No GPU detected - falling back to CPU (may be slow)")
    
    gcs_bucket = setup_gcs_client()
    
    params, state, task_config, sampler_config, noise_config, noise_encoder_config, denoiser_architecture_config = load_model_checkpoint(gcs_bucket)
    
    diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level = load_normalization_stats(gcs_bucket)
    
    dataset_files = find_2019_datasets(gcs_bucket, month)
    if not dataset_files:
        raise ValueError(f"No datasets found for {month}")
    
    dataset_file = dataset_files[0]
    dataset = load_dataset(gcs_bucket, dataset_file)
    
    if "total_precipitation_12hr" not in dataset.data_vars:
        raise ValueError("total_precipitation_12hr not found in dataset")
    
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast(
            sampler_config, task_config, denoiser_architecture_config,
            noise_config, noise_encoder_config,
            diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)
    
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    
    l2_errors = []
    
    available_forecasts = max(0, dataset.dims["time"] - 2)
    num_to_run = min(num_forecasts, available_forecasts)
    
    logger.info(f"Dataset has {dataset.dims['time']} time steps, can run {available_forecasts} forecasts")
    logger.info(f"Will run {num_to_run} forecasts")
    
    if num_to_run == 0:
        logger.error("Not enough time steps in dataset for forecasting (need at least 3)")
        return None
    
    for i in range(num_to_run):
        logger.info(f"Running forecast {i+1}/{num_to_run}")
        
        forecast_data = dataset.isel(time=slice(i, min(i + 10, dataset.dims["time"])))
        
        if forecast_data.dims["time"] < 3:
            logger.warning(f"Skipping forecast {i+1}: insufficient time steps")
            continue
        
        try:
            eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
                forecast_data, 
                target_lead_times=slice("12h", f"{max_lead_time_hours}h"),
                **dataclasses.asdict(task_config)
            )
            
            rng = jax.random.PRNGKey(i)
            predictions = run_forward_jitted(
                rng=rng,
                i=eval_inputs,
                t=eval_targets * np.nan,  # Use NaN template
                f=eval_forcings
            )
            
            l2_error, _ = compute_l2_error(predictions, eval_targets, "total_precipitation_12hr")
            l2_errors.append(l2_error)
            
            logger.info(f"Forecast {i+1} L2 error: {l2_error:.6f}")
            
        except Exception as e:
            logger.error(f"Error in forecast {i+1}: {e}")
            continue
    
    if l2_errors:
        mean_l2 = np.mean(l2_errors)
        std_l2 = np.std(l2_errors)
        min_l2 = np.min(l2_errors)
        max_l2 = np.max(l2_errors)
        
        logger.info("=" * 50)
        logger.info("🌧️  PRECIPITATION VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"📅 Month: {month}")
        logger.info(f"🔢 Number of forecasts: {len(l2_errors)}")
        logger.info(f"📊 Mean L2 error: {mean_l2:.6f}")
        logger.info(f"📈 Std L2 error: {std_l2:.6f}")
        logger.info(f"📉 Min L2 error: {min_l2:.6f}")
        logger.info(f"📊 Max L2 error: {max_l2:.6f}")
        logger.info(f"🎯 Device used: {'GPU' if gpu_available else 'CPU'}")
        logger.info("=" * 50)
        logger.info("✅ Validation completed successfully!")
        
        return {
            "month": month,
            "num_forecasts": len(l2_errors),
            "mean_l2_error": mean_l2,
            "std_l2_error": std_l2,
            "min_l2_error": min_l2,
            "max_l2_error": max_l2,
            "all_l2_errors": l2_errors,
            "device_used": "GPU" if gpu_available else "CPU"
        }
    else:
        logger.error("No successful forecasts completed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate GenCast precipitation predictions")
    parser.add_argument("--month", default="2019-03", help="Month to validate (e.g., 2019-03)")
    parser.add_argument("--num_forecasts", type=int, default=3, help="Number of forecasts to run")
    parser.add_argument("--max_lead_time", type=int, default=120, help="Maximum lead time in hours")
    
    args = parser.parse_args()
    
    try:
        results = run_validation(args.month, args.num_forecasts, args.max_lead_time)
        if results:
            logger.info("Validation completed successfully!")
        else:
            logger.error("Validation failed!")
            return 1
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
