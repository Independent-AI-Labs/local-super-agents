import os
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def concatenate_py_files(repo_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    try:
                        # Detect the encoding of the file
                        encoding = detect_encoding(file_path)
                        with open(file_path, 'r', encoding=encoding) as infile:
                            outfile.write(f"# File: {file_path}\n")
                            outfile.write(infile.read())
                            outfile.write("\n\n" + "#" * 79 + "\n\n")
                    except (UnicodeDecodeError, FileNotFoundError) as e:
                        print(f"Skipping file {file_path} due to encoding error: {e}")
                        # Try with utf-8
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                outfile.write(f"# File: {file_path}\n")
                                outfile.write(infile.read())
                                outfile.write("\n\n" + "#" * 79 + "\n\n")
                        except UnicodeDecodeError as e:
                            print(f"Failed to read file {file_path} with utf-8 encoding: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {file_path}: {e}")

if __name__ == "__main__":
    repo_path = input("Enter the path to your code repository: ")
    output_file = input("Enter the name of the output file (e.g., combined.py): ")

    concatenate_py_files(repo_path, output_file)
    print(f"All .py files have been concatenated into {output_file}")