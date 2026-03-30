"""
Dataset Downloader for Logistic Regression

This script downloads the 'Social_Network_Ads' dataset from Kaggle 
and places it completely within the local workspace ('./dataset').

Key functionalities:
1. Checks if the dataset is already present locally to avoid redundant downloads.
2. Uses `kagglehub` to fetch the raw dataset.
3. Moves the target CSV file directly into the local workspace directory.
4. Automatically cleans up the `kagglehub` cache directory so no residual 
   data is left anywhere else on the machine.
5. Performs a quick verification by printing the first few rows via pandas.
"""

import kagglehub
import os
import shutil
import pandas as pd

def main():
    # 1. Define your target destination first
    target_dir = "./dataset"
    target_file = os.path.join(target_dir, "Social_Network_Ads.csv")

    # 2. Check if the file already exists locally
    if os.path.exists(target_file):
        print(f"Dataset already exists at {target_file}. Skipping download...")
    else:
        print("Dataset not found locally. Starting download...")
        
        # 3. Download to the default cache
        cache_path = kagglehub.dataset_download("rakeshrau/social-network-ads")

        # 4. Create the directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 5. Locate the downloaded file and move it
        # * Note: sometimes kagglehub returns the file path directly, 
        # but usually it's a folder.
        if os.path.isdir(cache_path):
            downloaded_file = os.path.join(cache_path, "Social_Network_Ads.csv")
        else:
            downloaded_file = cache_path

        if os.path.exists(downloaded_file):
            shutil.move(downloaded_file, target_file)
            print(f"File successfully moved to: {target_file}")
            
            # Clean up the cache directory so no data is left behind
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path, ignore_errors=True)
            elif os.path.exists(cache_path):
                os.remove(cache_path)
            print("Cache cleaned up. Data is now only in your workspace.")
        else:
            print("Could not find Social_Network_Ads.csv in the downloaded cache.")

    # 6. A quick check (Always runs)
    if os.path.exists(target_file):
        df = pd.read_csv(target_file)
        print("\nQuick check - First 5 rows:")
        print(df.head())
    else:
        print("\nError: Could not load data because the file is missing.")

if __name__ == "__main__":
    main()