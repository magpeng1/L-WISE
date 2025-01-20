import os
import argparse
import pandas as pd

def generate_dirmap_csv(dataset_path, output_path=None, allowed_split_names=['train', 'test', 'val']):
    # Create an empty DataFrame
    df = pd.DataFrame(columns=["split", "class_num", "class", "im_path"])

    # Get the list of split directories
    split_dirs = [dir for dir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dir)) and dir in allowed_split_names]
    split_dirs.sort()

    # Iterate over each split directory
    for split in split_dirs:
        split_path = os.path.join(dataset_path, split)

        # Get the list of class directories
        class_dirs = [dir for dir in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, dir))]
        class_dirs.sort()

        # Assign integer values to each class
        class_map = {class_name: i for i, class_name in enumerate(class_dirs)}

        # Iterate over each class directory
        for class_name in class_dirs:
            class_path = os.path.join(split_path, class_name)

            # Get the list of image files in the class directory
            image_files = [file for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))]
            image_files.sort()

            # Create a DataFrame for the current class
            class_df = pd.DataFrame({
                "split": [split] * len(image_files),
                "class_num": [class_map[class_name]] * len(image_files),
                "class": [class_name] * len(image_files),
                "im_path": [os.path.join(split, class_name, file) for file in image_files]
            })

            # Append the class DataFrame to the main DataFrame
            df = pd.concat([df, class_df], ignore_index=True)

    # Sort the DataFrame by split, class_num, and im_path
    df = df.sort_values(by=["split", "class_num", "im_path"])

    # Save the DataFrame as a CSV file
    if output_path is None:
        output_path = os.path.join(dataset_path, "dirmap.csv")
    df.to_csv(output_path, index=False)

    print(f"Directory map saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a directory map CSV for an image dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the image dataset")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for the dirmap CSV file (optional)")
    args = parser.parse_args()

    generate_dirmap_csv(args.dataset_path, args.output_path)