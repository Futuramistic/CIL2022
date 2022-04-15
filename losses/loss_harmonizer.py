import numpy as np
import torch


DEFAULT_TORCH_DIM_LAYOUT = 'NCHW'


def torch_collapse_channel_dim(tensor, take_argmax, dim_layout=DEFAULT_TORCH_DIM_LAYOUT):
    shape = tensor.shape
    if len(shape) < 4:
        return tensor  # assume channel dim already collapsed
    channel_dim_idx = dim_layout.strip().upper().index('C')
    num_channels = shape[channel_dim_idx]
    if take_argmax and num_channels > 1:
        return torch.argmax(tensor, dim=channel_dim_idx).to(dtype=tensor.dtype)  # preserve original dtype
    else:
        # remove the channel dimension
        selector = [0 if dim_idx == channel_dim_idx else slice(0, shape[dim_idx]) for dim_idx in range(len(shape))]
        if take_argmax:
            return tensor[selector].round()
        else:
            return tensor[selector]


# TODO: add functionality of expanding to one channel only (effectively only "unsqueeze")?

def torch_expand_channel_dim(tensor, channel_starts=(0.0, 0.5), dim_layout=DEFAULT_TORCH_DIM_LAYOUT):
    # "channel_starts" simultaneously determines the number of channels to add, and the starting thresholds of these
    # channels
    channel_dim_idx = dim_layout.strip().upper().index('C')
    if len(tensor.shape) == len(channel_starts):
        tensor = torch_collapse_channel_dim(tensor, take_argmax=True)

    tensor = torch.unsqueeze(tensor, dim=channel_dim_idx)
    repeat_selector = [len(channel_starts) if dim_idx == channel_dim_idx else 1 for dim_idx in range(len(tensor.shape))]
    tensor = tensor.repeat(*repeat_selector)
    for channel_idx in range(len(channel_starts)):
        selector = [channel_idx if dim_idx == channel_dim_idx else slice(0, tensor.shape[dim_idx])
                    for dim_idx in range(len(tensor.shape))]
        target_channel = tensor[selector]
        if channel_idx == len(channel_starts) - 1:
            mask = channel_starts[channel_idx] <= target_channel
        else:
            mask = torch.logical_and(channel_starts[channel_idx] <= target_channel,
                                     target_channel < channel_starts[channel_idx + 1])
        tensor[selector] = mask.to(dtype=tensor.dtype)
    return tensor
