
import torch
import torch.nn.functional as F

class TPS:
    def __init__(self):
        """

        """
        pass

    def forward(self, X, Y, w, h, device):
        """
        :param X:  shape: n, k, 2
        :param Y:  shape: n, k, 2
        :param w:  image width
        :param h:  image height
        :param device: device
        :return: grid
        """

        n, k, _ = X.shape
        """ 计算grid"""
        grid = torch.ones((1, w, h, 2), dtype=torch.float32, device=device)
        grid[..., 0] = torch.arange(0, w, device=device)
        # torch.linspace(-1, 1, w, device=device)
        grid[..., 1] = torch.arange(0, h, device=device)[..., None]
        # torch.linspace(-1, 1, h, device=device)[..., None]
        grid = grid.reshape(-1, w* h, 2)  # shape: 1, w*h, 2

        """ 计算W, A"""
        distance = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)  # shape: n, k, k
        K = distance * torch.log(distance + 1e-15)  # shape: n, k, k
        K[torch.isnan(K)] = 0

        P = torch.cat((X, torch.ones((X.shape[0], X.shape[1], 1), dtype=torch.float32, device=device)),
                      dim=-1)  # shape: n, k, 3

        L_top = torch.cat((K, P), dim=-1)  # shape: n, k, k+3
        L_bottom = torch.cat((P.permute(0, 2, 1), torch.zeros((X.shape[0], 3, 3), dtype=torch.float32, device=device)),
                             dim=-1)  # shape: n, 3, k+3

        L = torch.cat((L_top, L_bottom), dim=1)  # shape: n, k+3, k+3
        Z = torch.cat((Y, torch.zeros((X.shape[0], 3, 2), dtype=torch.float32, device=device)),
                      dim=1)  # shape: n, k+3, 2
        Q = torch.linalg.solve(L, Z)  #

        """ 计算U """
        grid_distance = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)  # shape: n, w*h, k
        grid_K = grid_distance * torch.log(grid_distance + 1e-15)  # shape: n, w*h, k
        grid_K[torch.isnan(grid_K)] = 0

        P = torch.cat((grid, torch.ones((1, w * h, 1), dtype=torch.float32, device=device)), dim=-1)  # shape: 1, w*h, 3

        """ 计算warp grid """

        grid = grid_K @ Q[:, :k, :] + P @ Q[:, k:, :]  # shape: n, w*h, 2
        grid = grid.reshape(-1, w, h, 2)  # shape: n, w, h, 2

        return grid


import matplotlib

matplotlib.use("TkAgg")  # 或其他支持的后端，如 "Agg", "Qt5Agg", "GTK3Agg" 等。
import matplotlib.pyplot as plt
import cv2


def Norm(points, height, width):
    """
    :param points: shape: n, k, 2
    :return: shape: n, k, 2
    """
    points[..., 0] = points[..., 0] / width * 2 - 1
    points[..., 1] = points[..., 1] / height * 2 - 1
    return points


def scale(points, height, width):
    """
    :param points: shape: n, k, 2
    :return: shape: n, k, 2
    """
    points[..., 0] = (points[..., 0] + 1) / 2 * width
    points[..., 1] = (points[..., 1] + 1) / 2 * height
    return points


if __name__ == '__main__':
    image_size = [200, 200]
    image = torch.zeros(1, 1, image_size[0], image_size[1])
    image[:, :, 50:200, 50:200] = 255
    src_points = torch.FloatTensor([[60, 60], [140, 60], [140, 140], [60, 140]])
    dst_points = torch.FloatTensor([[50, 60], [130, 60], [100, 140], [20, 140]])

    # opencv版tps
    # tps = cv2.createThinPlateSplineShapeTransformer()
    #
    # matches = [cv2.DMatch(i, i, 0) for i in range(4)]
    # tps.estimateTransformation(src_points.view(1, -1, 2).numpy(), dst_points.view(1, -1, 2).numpy(), matches)
    # image_numpy = image[0, 0].numpy().astype('uint8')
    # img = tps.warpImage(image_numpy)

    # src_points = Norm(src_points, image_size[0], image_size[1])
    # dst_points = Norm(dst_points, image_size[0], image_size[1])
    #
    # print(src_points)
    # print(dst_points)
    #
    # print(src_points.shape)
    tps = TPS()
    grid = tps.forward(src_points[None, ...], dst_points[None, ...], image_size[0], image_size[1], 'cpu')

    grid = Norm(grid, image_size[0], image_size[1])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image[0, 0].numpy(), cmap='gray')

    img = F.grid_sample(image, grid, padding_mode='border')
    print(img.shape)

    plt.subplot(1, 2, 2)
    plt.title("Warped Image")
    plt.imshow(img[0, 0].numpy(), cmap='gray')
    plt.show()
