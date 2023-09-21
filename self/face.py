



import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

def face_detect(image_path = '/root/data/1_97.jpg', output_path = 'detected_faces.jpg'):
    # 加载图像
    image = cv2.imread(image_path)
    # OpenCV读取图像是BGR格式，转为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用dlib的面部检测器
    detector = dlib.get_frontal_face_detector()
    detections = detector(image_rgb)

    print("Number of faces detected: {}".format(len(detections)))
    # 打印形状、位置和大小
    # detection：矩形
    # left()：矩形左边的x坐标
    # top()：矩形顶部的y坐标
    # right()：矩形右边的x坐标
    # bottom()：矩形底部的y坐标
    for i, face in enumerate(detections):
        print('- Face #{}: Left: {} Top: {} Right: {} Bottom: {}'.format(
            i, face.left(), face.top(), face.right(), face.bottom()))

    # 为每个检测到的脸部绘制矩形
    for rect in detections:
        cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

    # 保存图像
    cv2.imwrite(output_path, image)


def get_dlib_masks(image, detector, expansion=(2, 1.5)):

    H, W, C = image.shape
    eh, ew = expansion

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 0)
    mask = np.zeros([len(rects), H, W], dtype=np.bool_)
    # mask = np.zeros([len(rects), H, W], dtype=np.uint8)

    for i, rect in enumerate(rects):

        # rect = rect.rect
        width = rect.right() - rect.left()
        height = rect.bottom() - rect.top()
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, rect.left() - dw)
        x2 = min(W-1, rect.right() + dw)
        y1 = max(0, rect.top() - dh)
        y2 = min(W-1, rect.bottom() + dh)

        mask[i, y1:y2, x1:x2] = True

    return mask



def visualize_and_save(image_path, save_path, detector):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = get_dlib_masks(image, detector)

    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # 显示所有掩码（可能有多个人脸）
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    plt.subplot(1, 2, 2)
    plt.imshow(combined_mask, cmap="gray")
    plt.title("Face Masks")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_dlib_mask(image_path, save_path, detector, expansion=(2, 1.5)):
    image = cv2.imread(image_path)
    mask = get_dlib_masks(image, detector, expansion)
    combined_mask = np.sum(mask, axis=0)
    binary_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
    print(binary_mask.shape)
    cv2.imwrite(save_path, binary_mask)
    print("Save mask to {}".format(save_path))



# 人体分割 ----

import dlib
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from detectron2.data import MetadataCatalog

def get_face_masks(image,
                   cfg_name="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                   dat_path="data/mmod_human_face_detector.dat",
                   save_path=None):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)
    predictor = DefaultPredictor(cfg)

    box_masks, seg_masks = get_detectron_masks(image, predictor, save_path=save_path)
    seg_mask = seg_masks.sum(axis=0) > 0
    # print(seg_mask.shape, box_masks.shape)
    return seg_mask, box_masks

def get_detectron_masks(image, predictor, classes=None, expansion=(1., 1.), save_path=None):

    H, W, C = image.shape

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    labels = instances.pred_classes.numpy()
    seg_masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    if classes is not None:
        indices = [i for i in range(len(instances)) if labels[i] in classes]
        seg_masks = seg_masks[indices]

    ew, eh = expansion
    boxes = np.round(boxes).astype(int)
    box_masks = np.zeros([len(instances), H, W], dtype=np.bool_)

    for i in range(len(instances)):
        if classes is not None and labels[i] not in classes: continue

        x1, y1, x2, y2 = boxes[i]

        width = x2 - x1
        height = y2 - y1
        dw = int(round((ew - 1.) * width / 2.))
        dh = int(round((eh - 1.) * height / 2.))

        x1 = max(0, x1 - dw)
        x2 = min(W - 1, x2 + dw)
        y1 = max(0, y1 - dh)
        y2 = min(W - 1, y2 + dh)

        box_masks[i, y1:y2, x1:x2] = True


    if save_path:
        # cfg = get_cfg()
        # Use COCO's metadata as default
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("coco_2017_train"), scale=1.2)
        visualized_output = v.draw_instance_predictions(instances)
        visualized_image = visualized_output.get_image()[:, :, ::-1]
        cv2.imwrite(save_path, visualized_image)

    return box_masks, seg_masks




if __name__ == '__main__':
    # 使用函数保存可视化结果
    # cfg = get_cfg()
    image_path = '/root/data/1_97.jpg'
    save_path = "human_.jpg"
    # detector = dlib.get_frontal_face_detector()
    # visualize_and_save(image_path, save_path, detector)
    # save_dlib_mask(image_path, save_path, detector)
    image = cv2.imread(image_path)
    seg, box = get_face_masks(image)
    print(seg.shape, box.shape)
    # 保存可视化结果
    print(seg)
    plt.imshow(seg, cmap='gray')
    plt.savefig(save_path)
    # cv2.imwrite(save_path, seg.astype(np.uint8))







