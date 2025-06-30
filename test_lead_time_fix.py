#!/usr/bin/env python3
"""
Test script to verify the lead time shape mismatch fix works correctly.
"""
import sys
sys.path.append('/home/ubuntu/graphcast')
import logging
from precipitation_validation import run_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lead_time_fix():
    """Test that different lead times work without shape mismatch errors."""
    logger.info("🧪 Testing lead time shape mismatch fix...")
    
    lead_times_to_test = [12, 24, 48]
    
    for lead_time in lead_times_to_test:
        logger.info(f"\n--- Testing lead time {lead_time}h ---")
        try:
            results = run_validation(
                month="2019-03",
                num_forecasts=1,  # Just one forecast for quick testing
                max_lead_time_hours=lead_time
            )
            
            if results and results.get("num_forecasts", 0) > 0:
                logger.info(f"✅ Lead time {lead_time}h: SUCCESS - L2 error: {results['mean_l2_error']:.6f}")
            else:
                logger.error(f"❌ Lead time {lead_time}h: FAILED - No successful forecasts")
                
        except Exception as e:
            logger.error(f"❌ Lead time {lead_time}h: FAILED - {e}")
    
    logger.info("\n🏁 Lead time testing completed!")

if __name__ == "__main__":
    test_lead_time_fix()
