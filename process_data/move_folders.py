import os
import argparse
from typing import List


def list_folders(directory: str) -> List[str]:
    """
    Returns a list containing the names of folders within the specified directory.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return []

    # List all items in the directory
    items = os.listdir(directory)

    # Filter out items that are not directories
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    return folders


def move_folders(source_directory: str, destination_directory: str) -> None:
    """
    Move folders from the source directory to the destination directory.
    """
    # Ensure the source directory exists
    if not os.path.exists(source_directory):
        print(f"The source directory {source_directory} does not exist.")
        return

    # Ensure the destination directory exists, create if it doesn't
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Get the list of folders in the source directory
    folders = list_folders(source_directory)
    for folder in folders:
        source_folder_path = os.path.join(source_directory, folder)
        destination_folder_path = os.path.join(destination_directory, folder)
        
        try:
            os.system(f'mv "{source_folder_path}" "{destination_folder_path}"')
        except Exception as e:
            print(f"Failed to move folder: {source_folder_path}. Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_directory', type=str, default="/Users/artemmerinov/data/holoassist/video_pitch_shifted")
    parser.add_argument('--destination_directory', type=str, default="/Users/artemmerinov/data/holoassist/HoloAssist")
    args = parser.parse_args()

    move_folders(args.source_directory, args.destination_directory)
