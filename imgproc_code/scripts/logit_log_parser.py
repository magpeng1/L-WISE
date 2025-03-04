#!/usr/bin/env python3
import re
import csv
import argparse
import os
import sys

def parse_log_file(log_file_path):
    """
    Parse the log file to extract epoch, train accuracy, and validation accuracy.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        list: List of dictionaries with 'epoch', 'train_acc', and 'val_acc' keys
    """
    results = []
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
    
    # Find all checkpoint blocks in the log file
    # Each block contains epoch number, train accuracy, and val accuracy
    checkpoint_blocks = re.findall(
        r'Processing checkpoint: (\d+)_checkpoint\.pt \(epoch: (\d+)\).*?'
        r'Processing split: train.*?'
        r'Accuracy: ([\d.]+).*?'
        r'Processing split: val.*?'
        r'Accuracy: ([\d.]+)',
        content, re.DOTALL
    )
    
    if not checkpoint_blocks:
        print("Warning: No matching checkpoint patterns found in the log file.")
    
    for block in checkpoint_blocks:
        file_num, epoch, train_acc, val_acc = block
        try:
            # Validate data before adding
            epoch_int = int(epoch)
            train_acc_float = float(train_acc)
            val_acc_float = float(val_acc)
            
            results.append({
                'epoch': epoch,
                'train_acc': train_acc_float,
                'val_acc': val_acc_float
            })
        except ValueError as e:
            print(f"Warning: Invalid data format for epoch {epoch}: {e}")
    
    # Sort by epoch number
    results.sort(key=lambda x: int(x['epoch']))
    
    return results

def write_csv(results, output_file):
    """
    Write the parsed results to a CSV file.
    
    Args:
        results (list): List of dictionaries with 'epoch', 'train_acc', and 'val_acc' keys
        output_file (str): Path to the output CSV file
    """
    if not results:
        print("Error: No results to write to CSV.")
        sys.exit(1)
    
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_acc', 'val_acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)

def main():
    """
    Main function to parse log file and generate CSV.
    
    Usage:
        python parse_log.py LOG_FILE_PATH [OUTPUT_CSV_PATH]
    """
    parser = argparse.ArgumentParser(
        description='Parse model checkpoint log file and create a CSV with accuracy metrics.'
    )
    parser.add_argument(
        'log_file', 
        help='Path to the log file'
    )
    parser.add_argument(
        '-o', '--output', 
        default='model_accuracy.csv',
        help='Path to the output CSV file (default: model_accuracy.csv)'
    )
    
    args = parser.parse_args()
    
    results = parse_log_file(args.log_file)
    write_csv(results, args.output)
    print(f"CSV file created: {args.output}")
    print(f"Processed {len(results)} checkpoint epochs.")

if __name__ == "__main__":
    main()