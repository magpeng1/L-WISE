import os
import cv2 as cv
import argparse

def get_args(argv=None):
  parser = argparse.ArgumentParser(description="Convert all images in a directory to PNG format recursively.")
  
  parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing images.")
  parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save images as PNG.")
  parser.add_argument('--ext', type=str, default=".png", help="extension to change the images to")
  parser.add_argument('--enforce_color', default=False, action='store_true', help="Save images as color even if they start out as grayscale (needed for compatibility with, e.g., MSRCR enhancement)")
  
  return parser.parse_args(argv)

def convert_to_ext(input_dir, output_dir, ext, enforce_color):
  valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

  assert ext.lower() in valid_extensions, f"Extension {ext} is not valid. Use one of the following: {valid_extensions}"
  
  for root, _, files in os.walk(input_dir):
    for file in files:
      if file.lower().endswith(valid_extensions):  # Check with case-insensitivity
        input_path = os.path.join(root, file)
        rel_path = os.path.relpath(input_path, input_dir)
        
        # Change the file extension to ext (e.g., .png, .JPEG)
        rel_path = os.path.splitext(rel_path)[0] + ext
        output_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read image
        img = cv.imread(input_path, cv.IMREAD_UNCHANGED)
        if img is None:
          print(f"Skipping {input_path}: Unable to read image.")
          continue

        if enforce_color and len(img.shape) == 2:  # Grayscale image
          img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        # Save with the intended extension
        cv.imwrite(output_path, img)
        print(f"Saved image as {ext} to {output_path}")
      else:
        print(f"Skipping non-image file: {file}")

def main(argv=None):
  args = get_args(argv)
  convert_to_ext(args.input_dir, args.output_dir, args.ext, args.enforce_color)

if __name__ == "__main__":
  main()
