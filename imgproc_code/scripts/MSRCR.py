## PLEASE NOTE: all input files to this script must be in color (3 channels, RGB), and must be .png files.

import os
import argparse
import subprocess
import cv2 as cv

def get_args(argv=None):
  parser = argparse.ArgumentParser(description="Apply Retinex algorithm on all images in a directory recursively.")
  
  parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing images.")
  parser.add_argument('--output_dir_rgb', type=str, required=True, help="Directory to save processed RGB images.")
  parser.add_argument('--output_dir_gray', type=str, required=True, help="Directory to save processed grayscale images.")
  parser.add_argument('--output_format', type=str, default='JPEG', help="Output image format (e.g., JPEG, PNG). Default: JPEG.")
  parser.add_argument('-S', type=int, default=3, help="Number of scales. Default: 3.")
  parser.add_argument('-L', type=int, default=15, help="Low scale. Default: 15.")
  parser.add_argument('-M', type=int, default=80, help="Medium scale. Default: 80.")
  parser.add_argument('-H', type=int, default=250, help="High scale. Default: 250.")
  parser.add_argument('-N', type=int, default=1, help="Final 'canonical' gain/offset or simplest color balance. Default: 1.")
  parser.add_argument('-l', type=int, default=1, help="Percentage of saturation on the left. Default: 1.")
  parser.add_argument('-R', type=int, default=1, help="Percentage of saturation on the right. Default: 1.")
  
  return parser.parse_args(argv)

def save_original_as(output_path, input_path, output_format):
  img = cv.imread(input_path, cv.IMREAD_UNCHANGED)
  if img is not None:
    output_file = os.path.splitext(output_path)[0] + f'.{output_format}'
    cv.imwrite(output_file, img)
    print(f"Saved original image as {output_format} to {output_file}")
  else:
    print(f"Error reading {input_path}. Could not save original image.")

def run_retinex_on_images(input_dir, output_dir_rgb, output_dir_gray, output_format, S, L, M, H, N, l, R):
  valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
  errors = []

  for root, _, files in os.walk(input_dir):
    for file in files:
      if file.lower().endswith(valid_extensions):
        input_path = os.path.join(root, file)
        rel_path = os.path.relpath(input_path, input_dir)

        # Set output paths for RGB and grayscale images
        output_rgb_path = os.path.join(output_dir_rgb, os.path.splitext(rel_path)[0] + f'.{output_format}')
        output_gray_path = os.path.join(output_dir_gray, os.path.splitext(rel_path)[0] + f'.{output_format}')

        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(output_rgb_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_gray_path), exist_ok=True)

        # Build the command string
        command = [
          "MSR_original/MSR_original",
          "-S", str(S),
          "-L", str(L),
          "-M", str(M),
          "-H", str(H),
          "-N", str(N),
          "-l", str(l),
          "-R", str(R),
          input_path,
          output_rgb_path,
          output_gray_path
        ]

        # Run the command
        try:
          subprocess.run(command, check=True)
          print(f"Processed RGB image saved to: {output_rgb_path}")
          print(f"Processed grayscale image saved to: {output_gray_path}")
        except subprocess.CalledProcessError as e:
          print(f"Error processing {input_path}: {e}")
          print("PLEASE NOTE: this can happen if the MSR utility is given a grayscale image. Use the --enforce_color option in to_png.py")
          errors.append(input_path)
          
          # Save the original image instead
          save_original_as(output_rgb_path, input_path, output_format)
          save_original_as(output_gray_path, input_path, output_format)

  # Summary of errors
  if errors:
    print(f"\nProcessing completed with {len(errors)} errors.")
    print("Files with errors:")
    for error_file in errors:
      print(error_file)
  else:
    print("Processing completed with no errors.")

def main(argv=None):
  args = get_args(argv)
  run_retinex_on_images(
    args.input_dir, 
    args.output_dir_rgb, 
    args.output_dir_gray, 
    args.output_format,
    args.S, 
    args.L, 
    args.M, 
    args.H, 
    args.N, 
    args.l, 
    args.R
  )

if __name__ == "__main__":
  main()
