import os

SUPPORTED_EXTENSIONS = [".py", ".java", ".cs", ".sql", ".c"]

def list_code_files(folder_path):
    code_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                code_files.append(os.path.join(root, file))
    return code_files

# Example usage:
code_folder = "./Files"
all_code_files = list_code_files(code_folder)

if not all_code_files:
    print("❌ No code files found in the directory.")
else:
    print("📁 Found code files:")
    for path in all_code_files:
        print(" -", path)
