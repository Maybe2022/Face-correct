



import cv2
import os

def frames_to_video(input_dir, output_video_path, fps=30, codec='mp4v'):
    """
    Convert frames from folders into a video.

    Parameters:
    - input_dir: Path to the root directory containing frame folders.
    - output_video_path: Path to save the output video.
    - fps: Frames per second for the output video.
    - codec: Codec to use for video compression. Default is 'mp4v' for .mp4 format.
    """

    # Get the list of frame folders
    frame_folders = [os.path.join(input_dir, f"frame_{idx+1}") for idx in range(len(os.listdir(input_dir)))]

    # Read the first frame to get the width and height
    first_frame_path = os.path.join(frame_folders[0], f"frame_1_output_temporal.jpg")
    frame = cv2.imread(first_frame_path)
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # Write each frame to the video
    for idx, frame_folder in enumerate(frame_folders):
        frame_path = os.path.join(frame_folder, f"frame_{idx+1}_output_temporal.jpg")
        img = cv2.imread(frame_path)
        out.write(img)

    # Release the VideoWriter
    out.release()

    print(f"Video saved at {output_video_path}")


# Example usage
frames_to_video(input_dir = "/root/test",output_video_path= 'output_video_temporal.mp4')
