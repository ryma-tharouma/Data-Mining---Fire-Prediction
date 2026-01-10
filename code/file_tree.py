import os

def list_files_recursive(path, indent=0):
    """Recursively list all files and folders in a given directory."""
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print("│   " * indent + f"📁 {item}/")
                list_files_recursive(item_path, indent + 1)
            else:
                size_kb = os.path.getsize(item_path) / 1024
                print("│   " * indent + f"📄 {item}  ({size_kb:.1f} KB)")
    except PermissionError:
        print("│   " * indent + "🚫 [Access Denied]")

# 🧭 Change this path to your main project/data folder
root_folder = r"."  # Example path
print(f"\nListing files in: {root_folder}\n{'='*60}")
list_files_recursive(root_folder)
