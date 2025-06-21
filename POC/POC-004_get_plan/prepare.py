import cv2
import numpy as np

def crop_and_resize(image, width, height, flip=False):
    """
    裁剪image到Width:Height的比例，然后缩放到Width*Height大小
    :param image: 输入图片，numpy数组，形状为(H, W, C)
    :param Width: 目标宽度
    :param Height: 目标高度
    :return: 裁剪并缩放后的图片
    """
    h, w = image.shape[:2]
    target_ratio = width / height
    current_ratio = w / h

    # 决定裁剪方式
    if current_ratio > target_ratio:
        # 图片太宽，裁剪左右
        new_w = int(h * target_ratio)
        start = (w - new_w) // 2
        image_cropped = image[:, start:start+new_w]
    else:
        # 图片太高，裁剪上下
        new_h = int(w / target_ratio)
        start = (h - new_h) // 2
        image_cropped = image[start:start+new_h, :]

    # 缩放到目标大小
    image_resized = cv2.resize(image_cropped, (width, height))
    if flip:
        image_resized = cv2.rotate(image_resized, cv2.ROTATE_180)
    return image_resized

def prepare(frame_rgb):
    frame_yuv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2YUV_I420)
    H = (frame_yuv.shape[0]*2)//3
    W = frame_yuv.shape[1]
    parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

    parsed[0] = frame_yuv[0:H:2, 0::2]
    parsed[1] = frame_yuv[1:H:2, 0::2]
    parsed[2] = frame_yuv[0:H:2, 1::2]
    parsed[3] = frame_yuv[1:H:2, 1::2]
    parsed[4] = frame_yuv[H:H+H//4].reshape((-1, H//2,W//2))
    parsed[5] = frame_yuv[H+H//4:H+H//2].reshape((-1, H//2,W//2))

    return parsed