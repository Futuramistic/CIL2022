from scipy.ndimage.measurements import label
import numpy as np
import tensorflow as tf
import torch


connection_filter = np.ones((3, 3), dtype=np.int32)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def remove_blobs(image, threshold=200):
    """
    Given a predicted groundtruth image, remove all the connected components whose size is below a certain threshold
    Args:
        image (numpy array, torch tensor or TF tensor): groundtruth image that we want to process
        threshold (int): Minimum size below which blobs are removed
    """
    original_shape = image.shape
    if threshold == 0: # No processing needed
        return image
    # Remove useless dimensions
    image = np.squeeze(image)
    is_tf = False
    is_torch = False
    # Convert to numpy arrays if not already done
    if tf.is_tensor(image):
        image = image.numpy().copy()
        is_tf = True
    elif torch.is_tensor(image):
        image = image.cpu().numpy()
        is_torch = True
    # Label the blobs
    labeled, ncomponents = label(image, connection_filter)
    indices = np.indices(image.shape).T[:, :, [1, 0]]
    # Remove the blobs that are not big enough
    for i in range(ncomponents):
        idcs = indices[labeled == i+1]
        component_size = len(idcs)
        if component_size < threshold:
            image[idcs[:, 0], idcs[:, 1]] = 0
    np.reshape(image, original_shape)
    # Convert back to original type if we made a modification
    if is_tf:
        image = tf.convert_to_tensor(image)
    elif is_torch:
        image = torch.Tensor(image).to(device)
    return image
