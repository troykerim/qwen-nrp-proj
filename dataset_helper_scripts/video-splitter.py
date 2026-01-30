'''
Process videos clips into individual images.
Creates 1 image per frame, which creates more images than 1 image per second

Note: 
If the video is 30 FPS, that means 30 frames per second & this script saves an image for every frame:

1 second of video → 30 images

1 minute (60 seconds) → 30 x 60s = 1,800 images

10 minutes → 18,000 images

'''

import cv2
import os

def extract_frames_from_folder(
    input_folder,
    output_folder,
    expected_width=1920,
    expected_height=1080
):
    """
    Extract EVERY frame from each .mp4 or .mov video in input_folder.
    Save frames as JPGs in output_folder.
    Only process videos that match the expected resolution.
    """

    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".mp4", ".mov")
    video_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    )

    if not video_files:
        print("No video clips found in input folder.")
        return

    print(f"Found {len(video_files)} video clips in {input_folder}\n")

    counter = 1  # Global image counter

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing: {video_file}")

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"  Could not open {video_file}. Skipping.\n")
            continue

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"  FPS: {fps:.2f}, Frames: {total_frames}, Resolution: {width}x{height}"
        )

        if width != expected_width or height != expected_height:
            print(
                f"  Skipping {video_file}: resolution mismatch ({width}x{height}).\n"
            )
            video.release()
            continue

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            filename = os.path.join(
                output_folder,
                f"#####-{counter}.jpg"   # RENAME HERE 
            )

            cv2.imwrite(filename, frame)
            counter += 1
            saved_count += 1
            frame_idx += 1

        video.release()
        print(f"  Extracted {saved_count} frames from {video_file}\n")

    print(f"Done! Extracted {counter - 1} frames total.")
    print(f"All frames saved in: {output_folder}")


if __name__ == "__main__":
    input_dir = ""
    output_dir = ""

    extract_frames_from_folder(
        input_dir,
        output_dir,
        expected_width=1920,
        expected_height=1080
    )
