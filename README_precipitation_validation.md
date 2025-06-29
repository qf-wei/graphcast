# GraphCast Precipitation Validation Program

This program validates the accuracy of GenCast precipitation predictions on RTX 5090 GPU by computing L2 error against ground truth data from 2019.

## Features

- ✅ **GPU-Compatible**: Uses `triblockdiag_mha` attention for RTX 5090 compatibility
- ✅ **Real Data**: Loads actual 2019 ERA5 weather data from Google Cloud Storage
- ✅ **L2 Error Computation**: Computes Root Mean Square Error for precipitation predictions
- ✅ **Memory Efficient**: Optimized for RTX 5090 GPU memory constraints
- ✅ **Comprehensive Stats**: Provides mean, std, min, max L2 error metrics

## Requirements

### Hardware
- NVIDIA RTX 5090 GPU
- At least 32GB system RAM
- Internet connection for dataset access

### Software
- Python 3.12+
- JAX with CUDA support: `pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- GraphCast dependencies (see main README)
- Google Cloud Storage client: `pip install google-cloud-storage`

## Usage

### Basic Usage
```bash
python precipitation_validation.py --month 2019-03 --num_forecasts 5
```

### Advanced Usage
```bash
python precipitation_validation.py \
    --month 2019-03 \
    --num_forecasts 10 \
    --max_lead_time 24
```

### Parameters
- `--month`: Month to validate (e.g., 2019-03) - currently only 2019-03 data available
- `--num_forecasts`: Number of forecasts to run (default: 3)
- `--max_lead_time`: Maximum lead time in hours (default: 120)

## Available Datasets

The program automatically discovers available 2019 datasets. Currently available:
- `source-era5_date-2019-03-29_res-0.25_levels-13_steps-01.nc` (and variants)
- `source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc` (and variants)

## Output

The program provides comprehensive validation results:

```
==================================================
🌧️  PRECIPITATION VALIDATION RESULTS
==================================================
📅 Month: 2019-03
🔢 Number of forecasts: 5
📊 Mean L2 error: 0.123456
📈 Std L2 error: 0.012345
📉 Min L2 error: 0.098765
📊 Max L2 error: 0.145678
🎯 Device used: GPU
==================================================
✅ Validation completed successfully!
```

## Technical Details

### Model Configuration
- Uses GenCast 1deg Mini model for RTX 5090 compatibility
- Attention type: `triblockdiag_mha` (GPU-compatible)
- Mask type: `full`
- Memory footprint: ~16GB vRAM, ~24GB system RAM

### L2 Error Computation
The L2 error is computed as Root Mean Square Error (RMSE):
```python
l2_error = sqrt(mean((prediction - target)^2))
```
Computed over spatial dimensions (lat, lon) for each time step.

### Data Processing
1. Loads ERA5 reanalysis data from Google Cloud Storage
2. Extracts inputs (2 time steps) and targets (1 time step)
3. Runs GenCast inference with proper normalization
4. Computes L2 error for `total_precipitation_12hr` variable

## Troubleshooting

### GPU Issues
If you see "No GPU detected", ensure:
- NVIDIA drivers are installed and up to date
- CUDA toolkit is properly installed
- JAX CUDA is installed: `pip install -U "jax[cuda12]"`

### Memory Issues
If you encounter OOM errors:
- Reduce `--num_forecasts` parameter
- Reduce `--max_lead_time` parameter
- Ensure no other GPU processes are running

### Dataset Issues
If datasets fail to load:
- Check internet connection
- Verify Google Cloud Storage access
- Try different month (currently only 2019-03 available)

## Files

- `precipitation_validation.py`: Main validation program
- `test_gpu.py`: GPU detection test script
- `check_datasets.py`: Dataset discovery script
- `check_models.py`: Model file discovery script

## Performance

Expected performance on RTX 5090:
- Model loading: ~3-5 seconds
- Per forecast: ~30-60 seconds (depending on lead time)
- Total validation (5 forecasts): ~3-5 minutes

## Validation Methodology

This validation follows standard meteorological practices:
1. Uses real historical data (ERA5 reanalysis)
2. Computes spatially-averaged RMSE
3. Provides statistical summary across multiple forecasts
4. Uses proper train/test split (model trained on 1979-2018, tested on 2019)
