import os
import shutil

root=r'D:\SD\training\MirrorSex\training'

def process_folder(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            # Recursively process subdirectories
            process_folder(file_path)
        else:
            # Copy non-directory files to root directory
            shutil.move(file_path,root)

def delete_folders(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            # Remove empty folders
            os.rmdir(file_path)

# Call the function to process the specified folder
process_folder(root)
# Call the function to delete the empty folders
delete_folders(root)
