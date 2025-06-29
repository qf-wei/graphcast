#!/usr/bin/env python3
"""
Test script to verify the precipitation validation program structure
without running the full inference (for environments without GPU).
"""

import logging
from precipitation_validation import (
    setup_gcs_client, 
    find_2019_datasets, 
    load_dataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_validation_structure():
    """Test the validation program structure without running inference."""
    logger.info("Testing validation program structure...")
    
    try:
        logger.info("1. Testing GCS connection...")
        gcs_bucket = setup_gcs_client()
        logger.info("✓ GCS connection successful")
        
        logger.info("2. Testing dataset discovery...")
        datasets = find_2019_datasets(gcs_bucket, "2019-03")
        if datasets:
            logger.info(f"✓ Found {len(datasets)} datasets")
        else:
            logger.error("✗ No datasets found")
            return False
        
        logger.info("3. Testing dataset loading...")
        dataset = load_dataset(gcs_bucket, datasets[0])
        logger.info(f"✓ Dataset loaded with shape: {dataset.dims}")
        
        if "total_precipitation_12hr" in dataset.data_vars:
            logger.info("✓ total_precipitation_12hr variable found")
        else:
            logger.error("✗ total_precipitation_12hr variable not found")
            return False
        
        logger.info("4. Testing model discovery...")
        from check_models import check_models
        models = check_models()
        if models:
            logger.info(f"✓ Found {len(models)} model files")
        else:
            logger.error("✗ No model files found")
            return False
        
        logger.info("=" * 50)
        logger.info("✅ All structure tests passed!")
        logger.info("The validation program should work on RTX 5090 GPU")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"✗ Structure test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_validation_structure()
    exit(0 if success else 1)
