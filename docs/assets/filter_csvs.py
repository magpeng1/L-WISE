import pandas as pd
import glob
import shutil
import os
from typing import Dict, List, Optional
import numpy as np

def make_backup_copies() -> List[str]:
    """Make backup copies of all CSV files in current directory."""
    csv_files = glob.glob("*.csv")
    for file in csv_files:
        base_name = file.rsplit('.', 1)[0]
        shutil.copy2(file, f"{base_name}_copy.csv")
    return csv_files

def filter_by_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Apply the filtering conditions and return filtered dataframe with removal counts."""
    initial_rows = len(df)
    removal_counts = {}
    
    """Please note: we elected to remove dermoscopy images of genital regions and female/unknown gender chest regions, ONLY for the illustrative purposes of the widget on our public website. 
    This is because many viewers may not be from medical backgrounds, and have not consented to view potentially confronting medical imagery. 
    We are aware that there is a problematic history of women being excluded from medical research, leading to health disparities. 
    We did not conduct any filtering of this kind for the research project itself, and informing human study participants about the nature of the images to be viewed was part of our informed consent process. 
    """
    # Filter for genital localization
    if "localization" in df.columns:
        genital_count = df[df["localization"] == "genital"].shape[0]
        df = df[df["localization"] != "genital"]
        removal_counts["genital_localization"] = genital_count
        
        # Filter for female/unknown gender chest conditions
        if "sex" in df.columns:
            chest_condition = (
                (df["sex"].isin(["female", "unknown"])) & 
                (df["localization"] == "chest")
            )
            chest_count = chest_condition.sum()
            df = df[~chest_condition]
            removal_counts["female_unknown_chest"] = chest_count
    
    # Handle duplicate lesion_ids
    if "lesion_id" in df.columns:
        duplicates = df.duplicated(subset=["lesion_id"], keep="first").sum()
        df = df.drop_duplicates(subset=["lesion_id"], keep="first")
        removal_counts["duplicate_lesion_ids"] = duplicates
    
    return df, removal_counts

def balance_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Balance classes by downsampling to the minority class size."""
    if "class" not in df.columns:
        return df, {}
    
    class_counts = df["class"].value_counts()
    min_class = class_counts.index[-1]
    min_count = class_counts.min()
    
    balanced_dfs = []
    for class_value in class_counts.index:
        class_df = df[df["class"] == class_value]
        if len(class_df) > min_count:
            # Preserve order while sampling
            class_df = class_df.sample(n=min_count, random_state=42)
            class_df = class_df.sort_index()
        balanced_dfs.append(class_df)
    
    return pd.concat(balanced_dfs), {
        "minority_class": min_class,
        "minority_class_count": min_count,
        "final_class_distribution": {
            class_value: min_count for class_value in class_counts.index
        }
    }

def calculate_difficulty(group):
    min_logit = group['robust_gt_logit'].min()
    max_logit = group['robust_gt_logit'].max()
    group['difficulty'] = 1 - ((group['robust_gt_logit'] - min_logit) / (max_logit - min_logit))
    return group

def process_csv_file(filename: str):
    """Process a single CSV file according to the specified conditions."""
    print(f"\nProcessing {filename}:")
    
    # Read the CSV
    df = pd.read_csv(filename)
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # Calculate normalized difficulty values from robust_gt_logit
    grouped = df.groupby('class')
    df = grouped.apply(calculate_difficulty)
    df.reset_index(drop=True, inplace=True)
    
    # Apply filters
    df, removal_counts = filter_by_conditions(df)
    
    # Print filtering summary
    for reason, count in removal_counts.items():
        if count > 0:
            print(f"Removed {count} rows due to {reason}")
    
    # Balance classes
    df, balance_info = balance_classes(df)
    
    if balance_info:
        print(f"\nClass balancing summary:")
        print(f"Minority class: {balance_info['minority_class']}")
        print(f"Instances per class: {balance_info['minority_class_count']}")
        print("\nFinal class distribution:")
        for class_value, count in balance_info['final_class_distribution'].items():
            print(f"Class {class_value}: {count} instances")
    
    # Save the processed file
    df.to_csv(filename, index=False)
    print(f"\nFinal row count: {len(df)}")
    print(f"Saved processed file: {filename}")

def main():
    """Main function to process all CSV files."""
    # Create backup copies
    csv_files = make_backup_copies()
    print(f"Created backup copies for {len(csv_files)} CSV files")
    
    # Process each CSV file
    for filename in csv_files:
        process_csv_file(filename)

if __name__ == "__main__":
    main()