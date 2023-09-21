


import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import cv2
import argparse
from torch import optim
from src.data import ImageDataset
from src.energy import Energy
from src.visualization import get_overlay_flow
import torch

parser = argparse.ArgumentParser(description='Distortion-Free-Wide-Angle-Portraits-on-Camera-Phones')
parser.add_argument('--file', type=str, required=False)

parser.add_argument('--num_iter', type=int, default=200, help="number of optimization steps")
parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
parser.add_argument('--Q', type=int, default=4, help="number of padding vertices")
parser.add_argument('--mesh_ds_ratio', type=int, default=20, help="the pixel-to-vertex ratio")

parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
parser.add_argument('--face_energy', type=float, default=4, help="weight of the face energy term")
parser.add_argument('--similarity', type=int, default=1, help="weight of similarity tranformation constraint")
parser.add_argument('--line_bending', type=float, default=4, help="weight of the line bending term")
parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")


def mesh_to_image(image, mesh_optimal):

    H, W, _ = image.shape
    map_optimal = cv2.resize(mesh_optimal, (W, H))
    x, y = map_optimal[:, :, 0] + W / 2, map_optimal[:, :, 1] + H / 2


    # # 设定扩展的边界宽度
    border_width = 1
    # 使用copyMakeBorder来扩展图像边界
    image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width,
                                      cv2.BORDER_REPLICATE)
    x = x + border_width
    y = y + border_width

    print("warping image")
    # map_optimal = cv2.resize(mesh_optimal, (W, H))
    # 接下来, 在padded的图像上进行remap
    image_corrected = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    # 最后, 裁剪掉扩展的边界
    out = cv2.resize(image_corrected[border_width : -border_width,border_width :-border_width,:], (W, H))
    return out


def process_image(file_name, args, dataset, mesh = None):
    print("loading {}".format(file_name))

    _, filename = os.path.split(file_name)
    filename, _ = os.path.splitext(filename)
    image, mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded = dataset.get_image_by_file(file_name)

    out_dir = os.path.dirname(file_name)  # 修改这里为输入图片的目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Your provided options setting here...
    if args.naive:
        trivial_mask = np.ones_like(correction_strength)
        box_masks_padded = trivial_mask[np.newaxis, :, :]
        seg_mask_padded = trivial_mask
        options = {
            "face_energy": 4,
            "similarity": False,
            "line_bending": 0,
            "regularization": 0,
            "boundary_constraint": 0
        }
    else:
        options = {
            "face_energy": args.face_energy,
            "similarity": args.similarity,
            "line_bending": args.line_bending,
            "regularization": args.regularization,
            "boundary_constraint": args.boundary_constraint
        }

    # load the optimization model
    print("loading the optimization model")
    model = Energy(options, mesh_uniform_padded, mesh_stereo_padded, correction_strength, box_masks_padded,
                   seg_mask_padded, args.Q)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_iter, eta_min=0)

    # perform optimization
    print("optimizing")
    for i in range(args.num_iter):
        optim.zero_grad()
        loss = model.forward()

        if mesh is not None:
            loss = loss + torch.mean((model.mesh - mesh) ** 2) * 4.0

        loss.backward()
        optim.step()
        with torch.no_grad():
            model.mesh[0:,:,0] = model.source_mesh[0:,:,0]
            model.mesh[0:,:,-1] = model.source_mesh[0:, :, -1]
            model.mesh[1:, 0, :] = model.source_mesh[1:, 0, :]
            model.mesh[1:, -1, :] = model.source_mesh[1:, -1, :]

    print("step {}, loss = {}".format(i, loss.item()))
    # calculate optical flow from the optimized mesh
    print("calculating optical flow")
    mesh_optimal = model.mesh.detach().cpu().numpy()
    mesh_optimal = mesh_optimal[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    out = mesh_to_image(image, mesh_optimal,)
    cv2.imwrite(os.path.join(out_dir, "{}_output_temporal.jpg".format(filename)), out)
    print("results saved in {}".format(out_dir))

    return model.mesh.detach()

# Main code
def main():
    args = parser.parse_args()
    dataset = ImageDataset(args)
    mesh = None
    for file_name in dataset.data_list:
        print(file_name)
        mesh = process_image(file_name, args, dataset, mesh)






if __name__ == '__main__':

    main()