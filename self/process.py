
import cv2
import os

# 读取视频文件
video_path = "/root/self/1695222002634341.mp4"
cap = cv2.VideoCapture(video_path)

# 确保视频文件被成功打开
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 如果视频长度为8秒，每秒的帧数为fps，那么总帧数应该为8*fps
# 所以我们要从这些帧中选出100帧
frames_indices = list(range(0, total_frames, total_frames // 250))[:250]

output_dir = "/root/test"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

count = 1
for idx in frames_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    # 为每一帧创建一个目录
    frame_dir = os.path.join(output_dir, f"frame_{count}")
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    frame_name = os.path.join(frame_dir, f"frame_{count}.jpg")
    cv2.imwrite(frame_name, frame)

    count += 1

cap.release()
