import zipfile
import os

# Files and folders necessary for the Kaggle environment
files_to_zip = ['guardian', 'server', 'client.py', 'models.py', 'openenv.yaml', '__init__.py']

# Folders to explicitly skip (like huge model checkpoints and caches)
ignore_folders = ['checkpoints', '__pycache__', 'outputs', 'data']

print("Creating strict Kaggle-compatible zip file...")

with zipfile.ZipFile('guardian_env_kaggle.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in files_to_zip:
        if os.path.isfile(item):
            # Enforce forward slashes for Kaggle (Linux)
            arcname = item.replace('\\', '/')
            zipf.write(item, arcname=arcname)
            print(f"Added: {arcname}")
            
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                # Filter out ignored folders
                dirs[:] = [d for d in dirs if d not in ignore_folders]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # Enforce Linux-style forward slashes for the internal ZIP paths
                    arcname = file_path.replace('\\', '/')
                    zipf.write(file_path, arcname=arcname)
                    print(f"Added: {arcname}")

print("\n✅ SUCCESS! 'guardian_env_kaggle.zip' created successfully.")
print("Upload this exact file to your Kaggle dataset. All forbidden characters are gone, and checkpoints are excluded.")
