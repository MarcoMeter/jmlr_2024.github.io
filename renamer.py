import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            parts = filename.split('_')
            if len(parts) > 4 and parts[-2].isdigit() and parts[-3] == "seed":
                new_name = '_'.join(parts[:-1]) + ".html"
                
                # Ensure the new name doesn't collide with an existing file
                base_name = '_'.join(parts[:-2])
                if os.path.exists(os.path.join(directory, new_name)):
                    print(f"Error: File {new_name} already exists!")
                    continue
                
                src = os.path.join(directory, filename)
                dst = os.path.join(directory, new_name)
                os.rename(src, dst)
                print(f"Renamed {filename} to {new_name}")

# Provide the path to your folder here
folder_path = "./results/"
rename_files_in_directory(folder_path)
