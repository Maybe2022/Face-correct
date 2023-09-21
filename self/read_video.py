
import os
import cv2
import glob

def merge_frames(frames_paths):
    frames = [cv2.imread(path) for path in frames_paths]
    if frames[0].shape[0] > frames[0].shape[1]:  # 竖屏
        merged_frame = cv2.hconcat(frames)
    else:  # 横屏
        merged_frame = cv2.vconcat(frames)
    return merged_frame

def process_video(data_dir, output_path):
    frame_dirs = sorted(glob.glob(os.path.join(data_dir, "*/")))

    # 读取第一帧来确定视频参数
    frame_patterns = ['*input.jpg', '*out.jpg', '*out1.jpg']
    first_frame_paths = sorted(glob.glob(os.path.join(frame_dirs[0], frame_patterns[0])) +
                               glob.glob(os.path.join(frame_dirs[0], frame_patterns[1])) +
                               glob.glob(os.path.join(frame_dirs[0], frame_patterns[2])))
    merged_first_frame = merge_frames(first_frame_paths)
    h, w, layers = merged_first_frame.shape
    size = (w, h)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, size)

    for frame_dir in frame_dirs:
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, frame_patterns[0])) +
                             glob.glob(os.path.join(frame_dir, frame_patterns[1])) +
                             glob.glob(os.path.join(frame_dir, frame_patterns[2])))
        merged_frame = merge_frames(frame_paths)
        out.write(merged_frame)

    out.release()

if __name__ == "__main__":
    data_directory = ".../data"  # 请将...替换为实际路径
    output_video_path = "merged_video.mp4"
    process_video(data_directory, output_video_path)
    print(f"Video saved at {output_video_path}")

