from utils import *
from data_handling import *
from models import *
from trainers import *
# various preprocessors
def channelwise_preprocessing_dataloader(dataloader, x, is_gt):
    if is_gt:
        return x[:1, :, :].float() / 255
    stats = DATASET_STATS[dataloader.dataset]
    x = x[:3, :, :].float()
    x[0] = (x[0] - stats['pixel_mean_0']) / stats['pixel_std_0']
    x[1] = (x[1] - stats['pixel_mean_1']) / stats['pixel_std_1']
    x[2] = (x[2] - stats['pixel_mean_2']) / stats['pixel_std_2']
    return x

# preprocesser "factory" by name
def get_preprocessing(preproc_name, dataloader):
    """Helper function to get a preprocessing function from a user defined preprocessing name

    Args:
        preproc_name (str): name of the preprocessing function
    """
    ppc = preproc_name.lower()
    
    if ppc == "range_zero_one":
        preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)
        return preprocessing
    elif dataloader.dataset in DATASET_STATS:
        return lambda x, is_gt: channelwise_preprocessing_dataloader(dataloader, x, is_gt)
    else:
        preprocessing = lambda x, is_gt: (x[:3, :, :].float() / 255.0) if not is_gt else (x[:1, :, :].float() / 255)
