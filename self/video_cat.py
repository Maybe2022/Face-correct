import cv2

def vertically_concatenate_videos(video_files, output_file):
    # 读取视频
    caps = [cv2.VideoCapture(video_file) for video_file in video_files]

    # 获取视频属性
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    fourcc = int(caps[0].get(cv2.CAP_PROP_FOURCC))

    # 为输出视频创建一个VideoWriter对象
    out_height = sum([int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps])
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, out_height))

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                return

            # 如果帧的宽度与第一个视频不同，则调整大小
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
            frames.append(frame)

        # 竖直堆叠帧
        combined_frame = cv2.vconcat(frames)
        out.write(combined_frame)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()




# 使用示例
video_files = ['output_video.mp4', 'output_video_correct.mp4', 'output_video_temporal.mp4']
output_file = 'combined_video.mp4'
vertically_concatenate_videos(video_files, output_file)
