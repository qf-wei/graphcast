#!/usr/bin/env python3
"""
Precipitation Validation Program for GraphCast/GenCast on RTX 5090 GPU (with local caching)

This program loads 2019 weather data, runs GenCast inference on RTX 5090 GPU,
and validates the accuracy of total_precipitation_12hr predictions by computing L2 error
against ground truth.

Key Features:
- Uses GPU-compatible attention configuration (triblockdiag_mha)
- Loads real 2019 ERA5 weather data from Google Cloud Storage OR local cache
- LOCAL CACHING: Downloads datasets once and reuses them for faster validation cycles
- Computes L2 error for precipitation predictions vs ground truth
- Handles memory efficiently for RTX 5090 GPU inference
- Provides comprehensive validation statistics

Usage:
    python precipitation_validation_with_cache.py --month 2019-03 --num_forecasts 5 --cache_dir ./datasets
    
    python precipitation_validation_with_cache.py --month 2019-03 --num_forecasts 5 --cache_dir ./datasets

Requirements:
- RTX 5090 GPU with CUDA support
- JAX with CUDA installation
- GraphCast dependencies
- Internet connection for initial dataset download
- ~20GB disk space for dataset cache
"""

import argparse
import dataclasses
import datetime
import logging
import os
import shutil
from pathlib import Path
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
from graphcast import autoregressive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_gcs_client():
    """Setup Google Cloud Storage client for accessing GraphCast data."""
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    return gcs_bucket


def download_file_if_needed(gcs_bucket, remote_path: str, local_path: Path) -> bool:
    """Download file from GCS if it doesn't exist locally."""
    if local_path.exists():
        logger.info(f"✓ Using cached file: {local_path}")
        return True
    
    logger.info(f"📥 Downloading {remote_path} to {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        blob = gcs_bucket.blob(remote_path)
        with open(local_path, 'wb') as f:
            blob.download_to_file(f)
        logger.info(f"✅ Downloaded successfully: {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {remote_path}: {e}")
        return False


def load_model_checkpoint_cached(gcs_bucket, cache_dir: Path, model_name: str = "GenCast 1p0deg Mini <2019.npz"):
    """Load GenCast model checkpoint with local caching."""
    logger.info(f"Loading model checkpoint: {model_name}")
    
    remote_path = f"gencast/params/{model_name}"
    local_path = cache_dir / "models" / model_name
    
    if not download_file_if_needed(gcs_bucket, remote_path, local_path):
        raise RuntimeError(f"Failed to download model checkpoint: {model_name}")
    
    with open(local_path, "rb") as f:
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


def load_normalization_stats_cached(gcs_bucket, cache_dir: Path):
    """Load normalization statistics with local caching."""
    logger.info("Loading normalization statistics...")
    
    stats_files = [
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc", 
        "stddev_by_level.nc",
        "min_by_level.nc"
    ]
    
    stats_data = {}
    for stat_file in stats_files:
        remote_path = f"gencast/stats/{stat_file}"
        local_path = cache_dir / "stats" / stat_file
        
        if not download_file_if_needed(gcs_bucket, remote_path, local_path):
            raise RuntimeError(f"Failed to download stats file: {stat_file}")
        
        with open(local_path, "rb") as f:
            stats_data[stat_file.replace('.nc', '')] = xarray.load_dataset(f).compute()
    
    return (stats_data['diffs_stddev_by_level'], 
            stats_data['mean_by_level'],
            stats_data['stddev_by_level'], 
            stats_data['min_by_level'])


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


def load_dataset_cached(gcs_bucket, cache_dir: Path, dataset_file: str) -> xarray.Dataset:
    """Load a specific dataset file with local caching."""
    logger.info(f"Loading dataset: {dataset_file}")
    
    remote_path = f"gencast/dataset/{dataset_file}"
    local_path = cache_dir / "datasets" / dataset_file
    
    if not download_file_if_needed(gcs_bucket, remote_path, local_path):
        raise RuntimeError(f"Failed to download dataset: {dataset_file}")
    
    with open(local_path, "rb") as f:
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

    predictor = autoregressive.Predictor(predictor)

    return predictor


def compute_l2_error(predictions: xarray.Dataset, targets: xarray.Dataset, variable: str = "total_precipitation_12hr") -> Tuple[float, xarray.DataArray]:
    """Compute L2 error between predictions and ground truth for a specific variable."""
    if variable not in predictions.data_vars:
        raise ValueError(f"Variable {variable} not found in predictions")
    if variable not in targets.data_vars:
        raise ValueError(f"Variable {variable} not found in targets")
    
    pred_var = predictions[variable]
    target_var = targets[variable]
    
    pred_data = jnp.array(pred_var.values)
    target_data = jnp.array(target_var.values)
    
    squared_diff = (pred_data - target_data) ** 2
    
    logger.info(f"pred_data shape: {pred_data.shape}, target_data shape: {target_data.shape}")
    logger.info(pred_data)
    logger.info("" + "="*50)
    logger.info(target_data)
    
    spatial_axes = (-2, -1)  # lat, lon dimensions
    l2_error_per_time = jnp.sqrt(jnp.mean(squared_diff, axis=spatial_axes))
    
    mean_l2_error = float(jnp.mean(l2_error_per_time))
    
    logger.info(f"L2 error for {variable}: {mean_l2_error:.6f}")
    
    l2_error_da = xarray.DataArray(
        l2_error_per_time,
        dims=pred_var.dims[:-2],  # Remove lat, lon dimensions
        coords={dim: pred_var.coords[dim] for dim in pred_var.dims[:-2]}
    )
    
    return mean_l2_error, l2_error_da


def get_cache_info(cache_dir: Path) -> dict:
    """Get information about cached files."""
    info = {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "total_size_gb": 0,
        "file_count": 0,
        "datasets": [],
        "models": [],
        "stats": []
    }
    
    if cache_dir.exists():
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = Path(root) / file
                size_mb = file_path.stat().st_size / (1024 * 1024)
                info["total_size_gb"] += size_mb / 1024
                info["file_count"] += 1
                
                if "dataset" in root:
                    info["datasets"].append({"name": file, "size_mb": size_mb})
                elif "model" in root:
                    info["models"].append({"name": file, "size_mb": size_mb})
                elif "stats" in root:
                    info["stats"].append({"name": file, "size_mb": size_mb})
    
    return info


def run_validation_cached(month: str, cache_dir: str, num_forecasts: int = 10, max_lead_time_hours: int = 120):
    """Run precipitation validation with local dataset caching."""
    cache_path = Path(cache_dir)
    
    logger.info(f"Starting precipitation validation for {month}")
    logger.info(f"Cache directory: {cache_path}")
    logger.info(f"JAX devices available: {jax.local_devices()}")
    
    cache_info = get_cache_info(cache_path)
    if cache_info["exists"]:
        logger.info(f"📁 Cache contains {cache_info['file_count']} files ({cache_info['total_size_gb']:.1f} GB)")
    else:
        logger.info("📁 Cache directory doesn't exist - will create and populate")
    
    gpu_available = any('gpu' in str(device).lower() for device in jax.local_devices())
    if gpu_available:
        logger.info("✓ GPU detected - using GPU acceleration")
    else:
        logger.warning("⚠ No GPU detected - falling back to CPU (may be slow)")
    
    gcs_bucket = setup_gcs_client()
    
    params, state, task_config, sampler_config, noise_config, noise_encoder_config, denoiser_architecture_config = load_model_checkpoint_cached(gcs_bucket, cache_path)
    
    diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level = load_normalization_stats_cached(gcs_bucket, cache_path)
    
    dataset_files = find_2019_datasets(gcs_bucket, month)
    if not dataset_files:
        raise ValueError(f"No datasets found for {month}")
    
    dataset_1deg = [f for f in dataset_files if "res-1.0" in f]
    dataset_025deg = [f for f in dataset_files if "res-0.25" in f]
    
    if dataset_1deg:
        dataset_file = dataset_1deg[0]
        logger.info(f"Using 1.0deg dataset to match 1deg model: {dataset_file}")
    elif dataset_025deg:
        dataset_file = dataset_025deg[0]
        logger.info(f"Using 0.25deg dataset (may cause shape mismatch): {dataset_file}")
    else:
        dataset_file = dataset_files[0]
        logger.info(f"Using first available dataset: {dataset_file}")
    
    dataset = load_dataset_cached(gcs_bucket, cache_path, dataset_file)
    
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
        
        final_cache_info = get_cache_info(cache_path)
        
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
        logger.info(f"💾 Cache size: {final_cache_info['total_size_gb']:.1f} GB ({final_cache_info['file_count']} files)")
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
            "device_used": "GPU" if gpu_available else "CPU",
            "cache_info": final_cache_info
        }
    else:
        logger.error("No successful forecasts completed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate GenCast precipitation predictions with local caching")
    parser.add_argument("--month", default="2019-03", help="Month to validate (e.g., 2019-03)")
    parser.add_argument("--num_forecasts", type=int, default=3, help="Number of forecasts to run")
    parser.add_argument("--max_lead_time", type=int, default=120, help="Maximum lead time in hours")
    parser.add_argument("--cache_dir", default="./graphcast_cache", help="Directory for local dataset cache")
    parser.add_argument("--clear_cache", action="store_true", help="Clear cache before running")
    
    args = parser.parse_args()
    
    if args.clear_cache and Path(args.cache_dir).exists():
        logger.info(f"🗑️ Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
    
    try:
        results = run_validation_cached(args.month, args.cache_dir, args.num_forecasts, args.max_lead_time)
        if results:
            logger.info("Validation completed successfully!")
            logger.info(f"💡 Next runs will be faster using cached data in: {args.cache_dir}")
        else:
            logger.error("Validation failed!")
            return 1
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
