from scipy.ndimage.measurements import label
import numpy as np
import cv2
import tensorflow as tf
import torch

connection_filter = np.ones((3, 3), dtype=np.int32)

# with_blobs = (cv2.imread('output_preds/satimage_151.png')[:, :, 0] / 255)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def remove_blobs(image, threshold=200):
    if threshold == 0:
        return image
    restore_first_dim = False
    restore_second_dim = False
    if len(image.shape) == 3:
        image = image[0]
        restore_first_dim = True
    elif len(image.shape) == 4:
        image = image[0][0]
        restore_second_dim = True
    print(image.shape)
    is_tf = False
    is_torch = False
    if tf.is_tensor(image):
        image = image.numpy()
        is_tf = True
    elif torch.is_tensor(image):
        image = image.cpu().numpy()
        is_torch = True

    labeled, ncomponents = label(image, connection_filter)
    indices = np.indices(image.shape).T[:, :, [1, 0]]
    for i in range (ncomponents):
        idcs = indices[labeled == i+1]
        component_size = len(idcs)
        if component_size < threshold:
            image[idcs[:, 0], idcs[:, 1]] = 0
    if restore_first_dim:
        image = image[None, :, :]
    elif restore_second_dim:
        image = image[None, None, :, :]
    if is_tf:
        image = tf.convert_to_tensor(image)
    elif is_torch:
        image = torch.Tensor(image).to(device)
    return image


# without_blobs = remove_blobs(with_blobs) * 255
# cv2.imwrite('without_blobs.png', without_blobs)

