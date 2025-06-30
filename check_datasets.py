#!/usr/bin/env python3
"""Check available 2019 datasets in GraphCast GCS bucket."""

from google.cloud import storage

def check_datasets():
    print("Checking available 2019 datasets...")
    
    try:
        gcs_client = storage.Client.create_anonymous_client()
        gcs_bucket = gcs_client.get_bucket('dm_graphcast')
        
        datasets_2019 = []
        print("Available 2019 datasets:")
        
        for blob in gcs_bucket.list_blobs(prefix='gencast/dataset/'):
            name = blob.name.removeprefix('gencast/dataset/')
            if name and '2019' in name:
                datasets_2019.append(name)
                print(f"  {name}")
                if len(datasets_2019) >= 20:
                    print("  ... (showing first 20)")
                    break
        
        print(f"\nTotal 2019 datasets found: {len(datasets_2019)}")
        
        jan_2019 = [d for d in datasets_2019 if '2019-01' in d or 'jan' in d.lower()]
        if jan_2019:
            print(f"January 2019 datasets: {jan_2019}")
        
        return datasets_2019
        
    except Exception as e:
        print(f"Error checking datasets: {e}")
        return []

if __name__ == "__main__":
    check_datasets()
