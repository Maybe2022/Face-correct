import os
import cv2
import torch
import numpy as np

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('path_to_your_model.pt')
model.to(device)
model.eval()


def process_image(img_path):
    img = cv2.imread(img_path)
    processed_img = model_predict(img).squeeze(0)
    return processed_img


def model_predict(data):
    tensor = torch.from_numpy(data).float().to(device)
    output = model(tensor)
    corrected_data = output.cpu().numpy().astype('uint8')
    return corrected_data


def process_video(video_path, output_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    while success:
        frames.append(image)
        success, image = vidcap.read()

    video_data = np.stack(frames, axis=0)
    corrected_video = model_predict(video_data)

    height, width, layers = corrected_video[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    for frame in corrected_video:
        out.write(frame)
    out.release()


from PIL import Image, ExifTags


def open_image_with_orientation(image_path):
    img = Image.open(image_path)

    try:
        # 获取图像的 Exif 数据
        exif = dict(img._getexif().items())

        # 如果 Exif 数据中存在方向标记
        if ExifTags.TAGS["Orientation"] in exif:

            orientation = exif[ExifTags.TAGS["Orientation"]]

            # 旋转图片
            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180)
            elif orientation == 4:
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # 有些图片可能没有 Exif 数据，直接返回原图片
        pass

    return img


from PIL import Image

def concatenate_images(img1, img2):
    # 确保两张图片高度一样
    assert img1.size[1] == img2.size[1], "两张图片的高度需要相同"

    # 创建一个新图片，宽度是两张图片的宽度之和，高度与原图相同
    combined_width = img1.size[0] + img2.size[0]
    combined_img = Image.new('RGB', (combined_width, img1.size[1]))

    # 将两张图片贴到新图上
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.size[0], 0))

    # 保存合并后的图片
    combined_img.save('combined_image.jpg')

# 读取两张图片
img1 = Image.open('path_to_image1.jpg')
img2 = Image.open('path_to_image2.jpg')

# 并排放置两张图片并保存
concatenate_images(img1, img2)


from PIL import Image

def concatenate_images(img_list, save_path):
    # 确定是横屏还是竖屏
    if img_list[0].size[0] > img_list[0].size[1]:  # 横屏
        # 确保所有图片宽度相同
        assert all([img.size[0] == img_list[0].size[0] for img in img_list]), "所有图片的宽度需要相同"

        # 创建一个新的图片，高度是所有图片的高度之和，宽度与原图相同
        combined_height = sum([img.size[1] for img in img_list])
        combined_img = Image.new('RGB', (img_list[0].size[0], combined_height))

        y_offset = 0
        for img in img_list:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.size[1]

    else:  # 竖屏
        # 确保所有图片高度相同
        assert all([img.size[1] == img_list[0].size[1] for img in img_list]), "所有图片的高度需要相同"

        # 创建一个新的图片，宽度是所有图片的宽度之和，高度与原图相同
        combined_width = sum([img.size[0] for img in img_list])
        combined_img = Image.new('RGB', (combined_width, img_list[0].size[1]))

        x_offset = 0
        for img in img_list:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]

    # 保存合并后的图片
    combined_img.save(save_path)

# 使用方法：
images = [Image.open('path_to_image1.jpg'), Image.open('path_to_image2.jpg'), Image.open('path_to_image3.jpg')]
save_path = 'combined_image.jpg'
concatenate_images(images, save_path)




if __name__ == "__main__":
    IMAGE_DIR = 'path_to_images_directory'
    VIDEO_DIR = 'path_to_videos_directory'
    OUTPUT_DIR = 'output_directory'

    for img_name in os.listdir(IMAGE_DIR):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(IMAGE_DIR, img_name)
            image = cv2.imread(image_path)
            image_data = np.transpose(image, (2, 0, 1))
            image_data = np.expand_dims(image_data, 0)
            corrected_img = process_image(image_data)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"corrected_{img_name}"), corrected_img)

    for video_name in os.listdir(VIDEO_DIR):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, video_name)
            output_video_path = os.path.join(OUTPUT_DIR, f"corrected_{video_name}")
            process_video(video_path, output_video_path)
