



import cv2
import numpy as np
import matplotlib.pyplot as plt


def FOV2f(fov, diagnoal):
    f = diagnoal / (2 * np.tan(fov / 2))
    return f


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

    epsilon = 1e-10
    ru = r0 * np.tan(0.5 * np.arctan(rp / f)) + epsilon

    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out


def get_uniform_stereo_mesh(image, fov, Q = 4,  mesh_ds_ratio = 40):

    H, W, _ = image.shape

    Hm = H // mesh_ds_ratio + 2 * Q
    Wm = W // mesh_ds_ratio + 2 * Q

    d = min(Hm, Wm) * mesh_ds_ratio
    f = FOV2f(fov, d)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))

    x = (np.arange(0, Wm, 1)).astype(np.float32) - (Wm // 2) + 0.5
    y = (np.arange(0, Hm, 1)).astype(np.float32) - (Hm // 2) + 0.5
    x = x * mesh_ds_ratio
    y = y * mesh_ds_ratio
    x, y = np.meshgrid(x, y)

    mesh_uniform = np.stack([x, y], axis=0)
    rp = np.linalg.norm(mesh_uniform, axis=0)
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x / ru * rp
    y = y / ru * rp
    mesh_stereo = np.stack([x, y], axis=0)

    return mesh_uniform, mesh_stereo



def correct_with_stereo_mesh(image, mesh_stereo):
    H, W, _ = image.shape

    # 将网格放大到与原图像相同的尺寸
    x_stereo_resized = cv2.resize(mesh_stereo[0], (W, H))
    y_stereo_resized = cv2.resize(mesh_stereo[1], (W, H))

    # 从立体网格到透视网格的映射
    map_x = np.clip(x_stereo_resized + W / 2, 0, W-1).astype(np.float32)
    map_y = np.clip(y_stereo_resized + H / 2, 0, H-1).astype(np.float32)

    # 使用 remap 函数纠正图像
    # corrected_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # 超出范围补黑

    corrected_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return corrected_image


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


if __name__ == '__main__':
    # 测试图片
    image = cv2.imread('/root/data/1_97.jpg')
    H,W,_ = image.shape
    x = (np.arange(0, W, 1)).astype(np.float32)
    y = (np.arange(0, H, 1)).astype(np.float32)
    x, y = np.meshgrid(x, y)
    mesh_uniform = np.stack([x, y], axis=0).transpose([1, 2, 0])
    print(mesh_uniform.shape)
    Wm = 100
    Hm = 100
    mesh_uniform = cv2.resize(mesh_uniform, (Wm, Hm))
    mesh_uniform = cv2.resize(mesh_uniform, (W, H))
    x, y = mesh_uniform[:,:,0], mesh_uniform[:,:,1]

    # 扩展图像边界
    pad = 50
    image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 调整网格坐标以匹配扩展后的图像
    x += pad
    y += pad

    image_corrected = cv2.remap(image_padded, x, y, interpolation=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    size = 10
    cv2.imwrite('test.jpg', cv2.resize(image_corrected[size:-size,size:-size:,:], (W, H)))

    # mesh_ds_ratio = 40
    # Q = 4
    # H, W, _ = image.shape
    # print(H, W)
    # Hm = H // mesh_ds_ratio + 2 * Q
    # Wm = W // mesh_ds_ratio + 2 * Q
    # x = (np.arange(0, Wm, 1)).astype(np.float32) - (Wm // 2) + 0.5
    # y = (np.arange(0, Hm, 1)).astype(np.float32) - (Hm // 2) + 0.5
    # x = (np.arange(0, Wm, 1)).astype(np.float32) / (Wm - 1) * 2 - 1
    # y = (np.arange(0, Hm, 1)).astype(np.float32) / (Hm - 1) * 2 - 1
    # x = x * mesh_ds_ratio
    # y = y * mesh_ds_ratio

    # x, y = np.meshgrid(x, y)
    # mesh_uniform = np.stack([x, y], axis=0)

    # mesh_optimal = mesh_uniform[:, Q:-Q, Q:-Q].transpose([1, 2, 0])


    # mesh_optimal = mesh_uniform.transpose([1, 2, 0])

    # 设定扩展的边界宽度
    # border_width = 80
    # 使用copyMakeBorder来扩展图像边界
    # image_padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width,
    #                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # print("image shape: ", image_padded.shape)
    # image_padded = image
    # H, W, _  = image_padded.shape
    #
    # print("warping image")
    # map_optimal = cv2.resize(mesh_optimal, (W, H),)
    # # [-1, 1] -> [0, W-1]
    # map_optimal[:, :, 0] = (map_optimal[:, :, 0] + 1) / 2 * (W - 1)
    # map_optimal[:, :, 1] = (map_optimal[:, :, 1] + 1) / 2 * (H - 1)
    # x, y = map_optimal[:, :, 0], map_optimal[:, :, 1]
    # # x, y = map_optimal[:, :, 0] + W // 2 , map_optimal[:, :, 1] + H // 2
    # print("x shape: ", x.shape)
    # # 接下来, 在padded的图像上进行remap
    # image_corrected_padded = cv2.remap(image_padded, x, y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # print("image_corrected_padded shape: ", image_corrected_padded.shape)
    # # 最后, 裁剪掉扩展的边界
    # # image_corrected = image_corrected_padded[border_width:-border_width, border_width:-border_width]
    # # image_corrected = image_corrected[20:-20, 20:-20]
    #
    # # image_corrected = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    #
    # cv2.imwrite('test.jpg', image_corrected_padded)

