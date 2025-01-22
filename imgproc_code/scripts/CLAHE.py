import os
import cv2 as cv
import argparse
import numpy as np

def get_args(argv=None):
  parser = argparse.ArgumentParser(description="Apply CLAHE to all images in a directory recursively.")
  
  parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing images.")
  parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save CLAHE-applied images.")
  parser.add_argument('--clip_limit', type=float, default=2.0, help="Clip limit parameter for CLAHE")
  
  return parser.parse_args(argv)

def apply_clahe(img, clip_limit):
  clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))

  if len(img.shape) == 3:  # Color image
    img_clahe = np.empty_like(img)
    for i in range(img.shape[2]):
      img_clahe[..., i] = clahe.apply(img[..., i])
  else:  # Grayscale image
    img_clahe = clahe.apply(img)
  
  return img_clahe

def process_images(input_dir, output_dir, clip_limit):
  valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
  
  for root, _, files in os.walk(input_dir):
    for file in files:
      if file.lower().endswith(valid_extensions):  # Check with case-insensitivity
        input_path = os.path.join(root, file)
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read image
        img = cv.imread(input_path, cv.IMREAD_UNCHANGED)
        if img is None:
          print(f"Skipping {input_path}: Unable to read image.")
          continue

        # Apply CLAHE
        img_clahe = apply_clahe(img, clip_limit)
        
        # Save processed image
        cv.imwrite(output_path, img_clahe)
        print(f"Saved CLAHE-applied image to {output_path}")

def main(argv=None):
  args = get_args(argv)
  process_images(args.input_dir, args.output_dir, args.clip_limit)

if __name__ == "__main__":
  main()
