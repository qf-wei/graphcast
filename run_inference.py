#!/usr/bin/env python3
"""Comprehensive inference script for GraphCast PyTorch model."""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple
import logging
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt

from graphcast import graphcast_torch
from graphcast import xarray_torch
from inference_pytorch import GraphCastInference, create_synthetic_data, create_synthetic_normalization_stats


def visualize_predictions(predictions: xr.Dataset, 
                        targets: Optional[xr.Dataset] = None,
                        output_dir: Path = Path("plots")):
    """Create visualizations of predictions."""
    output_dir.mkdir(exist_ok=True)
    
    for var_name in predictions.data_vars:
        var_data = predictions[var_name]
        
        if 'level' in var_data.dims:
            var_data = var_data.isel(level=0)
        
        if 'batch' in var_data.dims:
            var_data = var_data.isel(batch=0)
        
        if 'sample' in var_data.dims:
            var_data = var_data.mean('sample')
        
        if 'time' in var_data.dims and var_data.dims['time'] > 1:
            for t in range(min(4, var_data.dims['time'])):
                plt.figure(figsize=(12, 6))
                
                if len(var_data.dims) == 3:
                    data_slice = var_data.isel(time=t)
                    plt.subplot(1, 2, 1)
                    plt.imshow(data_slice.values, cmap='viridis')
                    plt.title(f'{var_name} - Prediction (t={t})')
                    plt.colorbar()
                    
                    if targets is not None and var_name in targets.data_vars:
                        target_data = targets[var_name]
                        if 'level' in target_data.dims:
                            target_data = target_data.isel(level=0)
                        if 'batch' in target_data.dims:
                            target_data = target_data.isel(batch=0)
                        if t < target_data.dims['time']:
                            plt.subplot(1, 2, 2)
                            plt.imshow(target_data.isel(time=t).values, cmap='viridis')
                            plt.title(f'{var_name} - Target (t={t})')
                            plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{var_name}_time_{t}.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        else:
            plt.figure(figsize=(8, 6))
            if len(var_data.dims) == 2:
                plt.imshow(var_data.values, cmap='viridis')
                plt.title(f'{var_name} - Prediction')
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(output_dir / f"{var_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    logging.info(f"Visualizations saved to {output_dir}")


def compute_metrics(predictions: xr.Dataset, targets: xr.Dataset) -> Dict[str, float]:
    """Compute prediction metrics."""
    metrics = {}
    
    for var_name in predictions.data_vars:
        if var_name in targets.data_vars:
            pred_data = xarray_torch.torch_data(predictions[var_name])
            target_data = xarray_torch.torch_data(targets[var_name])
            
            if pred_data.shape != target_data.shape:
                continue
            
            mse = torch.nn.functional.mse_loss(pred_data, target_data)
            mae = torch.nn.functional.l1_loss(pred_data, target_data)
            
            metrics[f'{var_name}_mse'] = mse.item()
            metrics[f'{var_name}_mae'] = mae.item()
            
            pred_std = pred_data.std().item()
            target_std = target_data.std().item()
            if target_std > 0:
                metrics[f'{var_name}_rmse_normalized'] = (mse.sqrt() / target_std).item()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run GraphCast inference')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with synthetic data')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--ensemble-size', type=int, default=4,
                       help='Number of ensemble members')
    parser.add_argument('--prediction-steps', type=int, default=8,
                       help='Number of prediction steps')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--compute-metrics', action='store_true',
                       help='Compute prediction metrics')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=('geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind'),
        target_variables=('geopotential', 'temperature'),
        forcing_variables=('2m_temperature',),
        pressure_levels=(500, 850, 1000),
        input_duration='12h'
    )
    
    if args.test_mode:
        logging.info("Running in test mode with synthetic data")
        
        model_config = graphcast_torch.ModelConfig(
            resolution=1,
            mesh_size=2,
            latent_size=64,
            gnn_msg_steps=4,
            hidden_layers=1,
            radius_query_fraction_edge_length=1.0
        )
        
        from graphcast.normalization_torch import InputsAndResiduals
        from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor
        
        base_model = graphcast_torch.GraphCast(model_config, task_config)
        
        stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
        normalized_model = InputsAndResiduals(
            base_model,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level
        )
        model = AutoregressivePredictor(normalized_model)
        
        inference_engine = GraphCastInference(model, device=device)
        
        inputs, targets_template, forcings = create_synthetic_data(task_config, args.prediction_steps)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logging.info("Running single-step prediction...")
        start_time = time.time()
        single_pred = inference_engine.predict_single_step(
            inputs, targets_template.isel(time=[0]), forcings.isel(time=[0])
        )
        single_time = time.time() - start_time
        logging.info(f"Single-step prediction completed in {single_time:.3f}s")
        
        logging.info("Running autoregressive prediction...")
        start_time = time.time()
        auto_pred = inference_engine.predict_autoregressive(
            inputs, targets_template, forcings
        )
        auto_time = time.time() - start_time
        logging.info(f"Autoregressive prediction completed in {auto_time:.3f}s")
        
        logging.info("Running ensemble prediction...")
        start_time = time.time()
        ensemble_pred = inference_engine.predict_ensemble(
            inputs, targets_template, forcings, 
            num_ensemble_members=args.ensemble_size
        )
        ensemble_time = time.time() - start_time
        logging.info(f"Ensemble prediction completed in {ensemble_time:.3f}s")
        
        logging.info("Running chunked prediction...")
        start_time = time.time()
        chunks = list(inference_engine.chunked_prediction(
            inputs, targets_template, forcings, num_steps_per_chunk=2
        ))
        chunked_pred = xr.concat(chunks, dim='time') if chunks else xr.Dataset()
        chunked_time = time.time() - start_time
        logging.info(f"Chunked prediction completed in {chunked_time:.3f}s")
        
        single_pred.to_netcdf(output_dir / "single_step_prediction.nc")
        auto_pred.to_netcdf(output_dir / "autoregressive_prediction.nc")
        ensemble_pred.to_netcdf(output_dir / "ensemble_prediction.nc")
        if chunked_pred:
            chunked_pred.to_netcdf(output_dir / "chunked_prediction.nc")
        
        if args.compute_metrics:
            logging.info("Computing prediction metrics...")
            single_metrics = compute_metrics(single_pred, targets_template.isel(time=[0]))
            auto_metrics = compute_metrics(auto_pred, targets_template)
            
            logging.info("Single-step metrics:")
            for key, value in single_metrics.items():
                logging.info(f"  {key}: {value:.6f}")
            
            logging.info("Autoregressive metrics:")
            for key, value in auto_metrics.items():
                logging.info(f"  {key}: {value:.6f}")
        
        if args.visualize:
            logging.info("Creating visualizations...")
            visualize_predictions(auto_pred, targets_template, output_dir / "plots")
            visualize_predictions(ensemble_pred.mean('sample'), targets_template, output_dir / "plots_ensemble")
        
        logging.info(f"Predictions saved to {output_dir}")
        logging.info("Inference completed successfully!")
        
    else:
        if args.checkpoint:
            logging.info(f"Loading model from checkpoint: {args.checkpoint}")
            inference_engine = GraphCastInference.load_from_checkpoint(
                Path(args.checkpoint), device=device
            )
            logging.info("Real data inference not fully implemented yet.")
        else:
            logging.error("Checkpoint required for real data inference. Use --test-mode for testing.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
