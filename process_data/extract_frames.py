import os
import argparse
import av
from PIL import Image


def extract_frames(
        holoassist_dir: str, 
        video_name: str,
        fps: int,
        width: int,
        height: int,
    ) -> None:
    """
    Extract frames from single video.
    """
    base_path = os.path.join(holoassist_dir, video_name, "Export_py")
    video_path = os.path.join(base_path, "Video_pitchshift.mp4")
    mpeg_img_path = os.path.join(base_path, "Video", "images")

    os.chdir(os.path.join(base_path, "Video"))
    if not os.path.exists(mpeg_img_path):
        # Export images if the path does not exist.
        os.mkdir(mpeg_img_path)

        command = (
            f"ffmpeg -i ../Video_pitchshift.mp4 "
            f"-vf 'fps={fps},scale={width}:{height}' "
            "-start_number 0 "
            "images/%06d.png"
            " > /dev/null 2>&1"
        )
        os.system(command)

        # # Open the video file
        # video = av.open(video_path)
        
        # # Get the video stream
        # video_stream = video.streams.video[0]
        # print(video_stream.average_rate)

        # # Calculate the interval for frame extraction
        # target_interval = int(video_stream.average_rate / fps)
        # print(target_interval)
        
        # img_num = 0
        # for frame_id, frame in enumerate(video.decode(video=0)):
        #     if frame_id % target_interval == 0:
        #         # Rescale the frame
        #         # frame = frame.reformat(width=width, height=height)
        #         # Convert the frame to an image
        #         # img = frame.to_image()
        #         img = frame.to_image().resize((width, height), Image.BILINEAR)
        #         # Save the image
        #         img.save(os.path.join(mpeg_img_path, f"{img_num:06d}.png"))
        #         img_num += 1


def extract_frames_all(
        holoassist_dir: str,
        fps: int,
        width: int,
        height: int,
    ) -> None:
    """
    Extract frames from all videos.
    """
    # List all items in the directory
    items = os.listdir(holoassist_dir)

    # Filter out items that are not directories
    video_names = [item for item in items if os.path.isdir(os.path.join(holoassist_dir, item))]

    for i, video_name in enumerate(video_names):
        extract_frames(
            holoassist_dir=holoassist_dir, 
            video_name=video_name,
            fps=fps, 
            width=width, 
            height=height,
        )
        print(f"{i:04d}/{len(video_names):04d} {video_name}")

    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--holoassist_dir', type=str, default="/Users/artemmerinov/data/holoassist/HoloAssist")
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=350)
    args = parser.parse_args()

    extract_frames_all(
        holoassist_dir=args.holoassist_dir,
        fps=args.fps, 
        width=args.width, 
        height=args.height,
    )