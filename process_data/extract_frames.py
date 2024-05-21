import os
import argparse
import time
from datetime import datetime


def extract_frames(
        holoassist_dir: str, 
        video_name: str,
        fps: int,
        width: int,
        height: int,
        threads: int,
    ) -> None:
    """
    Extract frames from a single video.
    """
    image_output_dir = os.path.join(holoassist_dir, video_name, "Export_py", "Video", "images")
    video_path = os.path.join(holoassist_dir, video_name, "Export_py", "Video_pitchshift.mp4")

    os.makedirs(image_output_dir, exist_ok=True)

    # Some video names has spaces, 
    # e.g. "R101- 2Aug-SmallPrinter", "R158-3Oct-RAM & Graphicscard",
    # therefore, we need to make proper command with inner quotes.

    command = (
        f'ffmpeg -i "{video_path}" '
        f'-threads {threads} '
        f'-vf "fps={fps},scale={width}:{height}" '
        f'-start_number 0 '
        f'"{os.path.join(image_output_dir, "%06d.png")}" '
        f'> /dev/null 2>&1'
    )
    os.system(command)


def extract_frames_all(
        holoassist_dir: str,
        fps: int,
        width: int,
        height: int,
        threads: int,
    ) -> None:
    """
    Extract frames from all videos.
    """
    # List all items in the directory
    items = os.listdir(holoassist_dir)

    # Filter out items that are not directories
    video_names = [item for item in items if os.path.isdir(os.path.join(holoassist_dir, item))]
    video_names = sorted(video_names)

    total_images_memory = 0

    for i, video_name in enumerate(video_names):
        extract_frames(
            holoassist_dir=holoassist_dir, 
            video_name=video_name,
            fps=fps, 
            width=width, 
            height=height,
            threads=threads,
        )
        
        # Print memory occupied by extracted images from a video
        image_output_dir = os.path.join(holoassist_dir, video_name, "Export_py", "Video", "images")
        
        images_memory = 0
        for image_file in os.listdir(image_output_dir):
            path_to_img = os.path.join(image_output_dir, image_file)
            images_memory += os.path.getsize(path_to_img)
        total_images_memory += images_memory

        print(f"progress={i:04d}/{len(video_names):04d}", 
              f"video_name={video_name}", 
              f"images_memory={images_memory / (1024 * 1024 * 1024):.2f} GB",
              f"total_images_memory={total_images_memory / (1024 * 1024 * 1024):.2f} GB",
              f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
              flush=True)
        
    print("Done.", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--holoassist_dir', type=str, default="/Users/artemmerinov/data/holoassist/HoloAssist")
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=350)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    extract_frames_all(
        holoassist_dir=args.holoassist_dir,
        fps=args.fps, 
        width=args.width, 
        height=args.height,
        threads=args.threads,
    )