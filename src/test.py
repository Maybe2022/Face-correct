
import cv2
import numpy as np



def pad_image_for_projection(image, fov):
    pad_size = 20
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT,)
    return padded_image, pad_size







def FOV2f(fov, d):
    return d / (2 * np.tan(fov / 2))

# def correct(image, fov):
#     h, w, _ = image.shape
#     d = min(h, w)
#     f = FOV2f(fov, d)
#     r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
#
#     x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
#     y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
#     x, y = np.meshgrid(x, y)
#
#     coords = np.stack([x, y], axis=-1)
#     rp = np.linalg.norm(coords, axis=-1)
#
#     eps = 1e-10
#     ru = r0 * np.tan(0.5 * np.arctan(rp / f)) + eps
#
#     x = x / ru * rp + w / 2
#     y = y / ru * rp + h / 2
#
#     out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)
#
#     return out

def correct(image, fov):

    h, w, _ = image.shape
    d = min(h, w)
    f = FOV2f(fov, d)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y)

    coords = np.stack([x, y], axis=-1)
    rp = np.linalg.norm(coords, axis=-1)
    ru = r0 * np.tan(0.5 * np.arctan(rp / f)) + 1e-10

    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out

# Test the function
# image = cv2.imread("/root/data/1_97.jpg")
# fov = 97
# padded_image,pad_size = pad_image_for_projection(image, fov)
# corrected_img = correct(padded_image, fov * np.pi / 180)[pad_size:-pad_size, pad_size:-pad_size, :]
# cv2.imwrite("corrected_image.jpg", corrected_img)


def mesh_to_image(image, mesh_optimal):

    H, W, _ = image.shape
    map_optimal = cv2.resize(mesh_optimal, (W, H))
    x, y = map_optimal[:, :, 0] + W / 2, map_optimal[:, :, 1] + H / 2


    # # 设定扩展的边界宽度
    border_width = 20
    # 使用copyMakeBorder来扩展图像边界
    image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width,
                                      cv2.BORDER_REPLICATE)


    print("warping image")
    # map_optimal = cv2.resize(mesh_optimal, (W, H))
    # 接下来, 在padded的图像上进行remap
    image_corrected = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    # 最后, 裁剪掉扩展的边界
    out = cv2.resize(image_corrected[10:-10,10:-10,:], (W, H))
    return out
