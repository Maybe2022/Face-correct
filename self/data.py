





# 自己复现的数据集加载器
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import dlib
import numpy as np
from mesh import get_uniform_stereo_mesh


class FaceDataset(Dataset):
    def __init__(self, image_paths, face_detector, segmenter, focal_length, fov, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.face_detector = face_detector
        self.segmenter = segmenter
        self.focal_length = focal_length
        self.fov = fov

    def __len__(self):
        return len(self.image_paths)

    def stereographic_projection(self, point, focal_length):
        rp = np.linalg.norm(point)
        ru = focal_length * np.tan(0.5 * np.arctan(rp / focal_length))
        scaling = ru / rp if rp > 1e-8 else 1
        return point * scaling

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # Detect face
        faces = self.face_detector(image)

        # Compute the mask for faces and hair using segmenter
        mask = self.segmenter(image)

        # Define source mesh (perspective projection, which is just a regular grid for our purposes)
        # Compute target mesh using stereographic projection
        source_mesh, target_mesh = get_uniform_stereo_mesh(image, self.fov, 0, 1)

        sample = {
            'image': image,
            'mask': mask,
            'faces': faces,
            'focal_length': self.focal_length,
            'source_mesh': source_mesh,
            'target_mesh': target_mesh
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# Define transformations and normalization (if necessary)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    # other transformations if needed
])

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Initialize the segmenter (This is just a placeholder, you'd replace it with the actual model)
# segmenter =  # ... initialize your segmenter model
#
# focal_length =  # ... set your focal length
# fov =  # ... set your field of view
#
# dataset = FaceDataset(image_paths=['path_to_image1', 'path_to_image2'], face_detector=detector, segmenter=segmenter,
#                       focal_length=focal_length, fov=fov, transform=data_transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
