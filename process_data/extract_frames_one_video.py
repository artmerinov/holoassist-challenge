from extract_frames import extract_frames
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--holoassist_dir', type=str, default="/Users/artemmerinov/data/holoassist/HoloAssist")
    parser.add_argument('--video_name', type=str, default="R071-19July-BigPrinter")
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=350)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()

    extract_frames(
        holoassist_dir=args.holoassist_dir,
        video_name=args.video_name,
        fps=args.fps, 
        width=args.width, 
        height=args.height,
        threads=args.threads,
    )