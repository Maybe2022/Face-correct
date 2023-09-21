import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from stereographic import get_uniform_stereo_mesh
from perception import get_face_masks, get_object_masks


# class ImageDataset(Dataset):
#
#     def __init__(self, args, root='data'):
#
#         self.Q = args.Q
#         self.mesh_ds_ratio = args.mesh_ds_ratio
#         self.data_list = []
#         for names in os.listdir(root):
#             if names.endswith(".jpg"):
#                 self.data_list.append(os.path.join(root, names))
#         self.data_list = sorted(self.data_list)
#
#
#     def get_image_by_file(self, file, classes=None):
#
#         data_name = file
#         # fov = int(data_name.split('/')[-1].split('.')[0].split('_')[-1])
#         fov = 90
#
#         image = cv2.imread(data_name)
#         H, W, _ = image.shape
#
#         Hm = H // self.mesh_ds_ratio
#         Wm = W // self.mesh_ds_ratio
#
#         if classes is None:
#             seg_mask, box_masks = get_face_masks(image)
#         else:
#             seg_mask, box_masks = get_object_masks(image, classes=classes)
#
#         seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
#         box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
#         box_masks = np.stack(box_masks, axis=0)
#         seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
#         box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")
#
#         mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, fov * np.pi / 180, self.Q, self.mesh_ds_ratio)
#
#         radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
#         half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
#         ra = half_diagonal / 2.
#         rb = half_diagonal / (2 * np.log(99))
#         correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))
#
#         return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded
#
#
#
#     def __getitem__(self, index):
#
#         index = index % len(self.data_list)
#         data_name = self.data_list[index]
#
#         return self.get_image_by_file(data_name)
#
#
#
#     def __len__(self):
#         return len(self.data_list)

import re
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]




class ImageDataset(Dataset):

    def __init__(self, args, root='/root/test'):

        self.Q = args.Q
        self.mesh_ds_ratio = args.mesh_ds_ratio
        self.data_list = []
        # for names in os.listdir(root):
        #     if names.endswith(".jpg"):
        #         self.data_list.append(os.path.join(root, names))
        for dir_name in os.listdir(root):
            dir_path = os.path.join(root, dir_name)
            if os.path.isdir(dir_path) and dir_name.startswith("frame_"):
                specific_file_name = f"{dir_name}.jpg"
                if specific_file_name in os.listdir(dir_path):
                    self.data_list.append(os.path.join(dir_path, specific_file_name))

        self.data_list = sorted(self.data_list, key=natural_sort_key)
        # self.data_list = sorted(self.data_list)

    def save_masks(self, filename, seg_mask, box_masks):
        np.savez(filename, seg_mask=seg_mask, box_masks=box_masks)

    def load_masks(self, filename):
        with np.load(filename) as data:
            seg_mask = data['seg_mask']
            box_masks = data['box_masks']
        return seg_mask, box_masks


    def get_image_by_file(self, file, classes=None):

        data_name = file
        # mask_filename = data_name.replace('.jpg', '_masks.npz')
        mask_filename = os.path.join(os.path.dirname(data_name), data_name.split('/')[-1].replace('.jpg', '_masks.npz'))

        # fov = int(data_name.split('/')[-1].split('.')[0].split('_')[-1])
        fov = 108

        image = cv2.imread(data_name)
        H, W, _ = image.shape

        Hm = H // self.mesh_ds_ratio
        Wm = W // self.mesh_ds_ratio

        if os.path.exists(mask_filename):
            # Load the masks from file
            seg_mask, box_masks = self.load_masks(mask_filename)
        else:
            if classes is None:
                seg_mask, box_masks = get_face_masks(image)
            else:
                seg_mask, box_masks = get_object_masks(image, classes=classes)

            # 检查 seg_mask 是否为空或形状不正确
            if seg_mask is None or seg_mask.shape != (H, W):
                seg_mask = np.zeros((H, W), dtype=np.float32)

            # 检查 box_masks 是否为空
            if len(box_masks) == 0:
                box_masks = [np.zeros((H, W), dtype=np.float32)]
            self.save_masks(mask_filename, seg_mask, box_masks)


        seg_mask = cv2.resize(seg_mask.astype(np.float32), (Wm, Hm))
        box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm, Hm)) for box_mask in box_masks]
        box_masks = np.stack(box_masks, axis=0)
        seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
        box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")

        mesh_uniform_padded, mesh_stereo_padded = get_uniform_stereo_mesh(image, fov * np.pi / 180, self.Q, self.mesh_ds_ratio)


        radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
        half_diagonal = np.linalg.norm([H + 2 * self.Q * self.mesh_ds_ratio, W + 2 * self.Q * self.mesh_ds_ratio]) / 2.
        ra = half_diagonal / 2.
        rb = half_diagonal / (2 * np.log(99))
        correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))

        return image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded



    def __getitem__(self, index):

        index = index % len(self.data_list)
        data_name = self.data_list[index]

        return self.get_image_by_file(data_name)


    def __len__(self):
        return len(self.data_list)


# 你好，帮我修改一些这个数据集代码：
# 要求：我的数据目录是这样的：/data/video/frame/...
# 1.我想把这个数据集改成读取视频的，每次读取一个视频，然后把视频的每一帧都读取出来，然后再把每一帧都进行处理。
# 2.每个帧文件夹里面的数据有：image.jpg，mask.png(人像mask)，face_bounding_box.(人脸框坐标)
# 替换这个代码，改成读本地文件的：        if classes is None:
#             seg_mask, box_masks = get_face_masks(image)
#         else:
#             seg_mask, box_masks = get_object_masks(image, classes=classes)




class args():
    def __init__(self):
        self.Q = 1
        self.mesh_ds_ratio = 20

if __name__ == '__main__':
    # image_path = '/root/data/1_97.jpg'
    # args = args()
    # dataset = ImageDataset(args = args)
    #
    # image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded = dataset.get_image_by_file(image_path)
    dataset = ImageDataset(args = args())
    print(len(dataset))
    # print(image.shape)
    # print(mesh_uniform_padded.shape)
    # print(correction_strength.shape)
    # print(seg_mask_padded.shape)
    # print(box_masks_padded.shape)
