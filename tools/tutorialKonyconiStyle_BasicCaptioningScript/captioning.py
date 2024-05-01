import os

def process_folder(path, word):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            # Recursively process subdirectories
            process_folder(file_path, word)
        elif file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Create a new file with the .txt extension
            txt_file_path = os.path.splitext(file_path)[0] + '.txt'
            # Get the folder name
            folder_name = os.path.basename(path)
            # Create content for the new file
            content = f'{word} {folder_name}\n'
            # Write the content to the new file
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(content)

# Call the function to process the specified folder
process_folder(r'D:\SD\training\training', 'MirrorWorld')
