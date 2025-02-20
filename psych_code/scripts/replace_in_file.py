import argparse

def replace_in_file(file_path, old_str, new_str):
    # Read the entire content of the file
    with open(file_path, 'r', newline='') as file:
        content = file.read()

    # Replace occurrences of the old string with the new string
    modified_content = content.replace(old_str, new_str)

    # Write the modified content back to the file
    with open(file_path, 'w', newline='') as file:
        file.write(modified_content)

def main():
    parser = argparse.ArgumentParser(description='Replace strings in a text or csv file.')
    parser.add_argument('--path', required=True, help='Path to the file')
    parser.add_argument('--replace', required=True, help='String to be replaced')
    parser.add_argument('--with', dest='replace_with', required=True, help='String to replace with')
    args = parser.parse_args()

    replace_in_file(args.path, args.replace, args.replace_with)
    print(f"Replacement complete in file: {args.path}")

if __name__ == '__main__':
    main()

