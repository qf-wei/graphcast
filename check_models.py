#!/usr/bin/env python3
"""Check available GenCast model files in GCS bucket."""

from google.cloud import storage

def check_models():
    print("Checking available GenCast model files...")
    
    try:
        gcs_client = storage.Client.create_anonymous_client()
        gcs_bucket = gcs_client.get_bucket('dm_graphcast')
        
        model_files = []
        print("Available GenCast model files:")
        
        for blob in gcs_bucket.list_blobs(prefix='gencast/params/'):
            name = blob.name.removeprefix('gencast/params/')
            if name:
                model_files.append(name)
                print(f"  {name}")
        
        print(f"\nTotal model files found: {len(model_files)}")
        return model_files
        
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

if __name__ == "__main__":
    check_models()
