# 💡 Ideas & links for the project

__Note__: many more good ideas can be found in the paper summaries; it might be worth going over the summaries and putting some ideas here

## 📂 Datasets:
* subdivide into patches ("slicing")
  * add padding & average predictions in overlapping patch regions
* more training data:
  * can adjust to necessary GSD (ground sampling distance) by down-/upsampling?
  * use other datasets
    * Massachusets roads DS: https://www.kaggle.com/elmirakhajei/massachusets-dataset
    * EPFL road segmentation DS: https://github.com/mukel/epfml17-segmentation/tree/master/data
    * DeepGlobe road segmentation DS (?): https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset
  * pull data using Google Maps API: https://developers.google.com/maps/documentation/maps-static/start
  * use existing images 
    * scale
    * rotate
    * mirror
    * change illumination/lighting
  * generate synthetic dataset using GAN: https://ieeexplore.ieee.org/document/9173823

## 🤖 Model architectures:
* Global context-based automatic road segmentation via dilated CNN: https://www.sciencedirect.com/science/article/abs/pii/S0020025520304862
* add dilated convolutions to models without dilation
* Deep Layer Aggregation (and/or Hierarchical Deep Aggregation?): https://arxiv.org/abs/1707.06484
* U-Net++, augmented U-Net
  * try fusing U-Net++ & augmented U-Net
* GL-Dense-U-Net: https://www.mdpi.com/2072-4292/10/9/1461
  * also try DeepLabv3 (?)
* can copy & adjust architectures from related fields (e.g. object detection)
  * __large list of satellite imagery-related papers & datasets at https://github.com/chrieke/awesome-satellite-imagery-datasets__
* get ideas from Prof. Yu's CV slides: https://moodle-app2.let.ethz.ch/pluginfile.php/1217519/mod_folder/content/0/ComputerVision2021-lecture6.pdf
* "rotated bounding box" idea
  * output proposal boxes similar to R-CNN (https://arxiv.org/abs/1311.2524), but additionally output rotation angle of box
    * check if there are papers doing this

## 📉 Loss functions:
* add regularizer encouraging "realistic" segmentation
  * e.g. total variation from Reliable & Trustworthy AI
* (focal) Tversky loss?
* countering class imbalance (road vs. BG) in "naive" losses
  * weighted MSE-based loss as in https://arxiv.org/abs/1802.01445

## 🛠️ Hyperparameter tuning:
* Hyperopt: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/IntroductionToHyperopt.ipynb
* log metrics (F1) & losses in MLflow (note HTTP protocol; http://algvrithm.com:8000/)
* upload model code snapshot (only relevant .py/notebook files) as artifact using MLflow before training: https://colab.research.google.com/drive/1dompA_rfWp3u3ngvSakaReXoUFx5t9ME#scrollTo=0BxTV4rHuDNS
  * will help reproduce earlier results later when code has changed
* parallelize using vast.ai if necessary

## 💻 Frameworks & coding:
* use TensorFlow
* use Conda
* choose Python version compatible with matplotlib (https://matplotlib.org/stable/devel/min_dep_policy.html#list-of-dependency-min-versions; seems to need Python 3.7)
* use .py files rather than notebooks; can keep one or few "front-end notebooks" e.g. for Google Colab
* document code well (will get reviewed)
* keep requirements.txt (first person who sets up Python should create it)
* for baselines, can use/adapt existing repos 
  * __warning__: make sure to give mark copied parts, give appropriate credit and consider the project's license!
  * be critical of "non-official" implementations (not implemented by authors themselves)
* consistency reasons might require us to reimplement baselines without official codebases, to evaluate all baseline models on our dataset using F1 score as metric


## Next TODOs
- add print command
- mask_to_submission to upload result
- find limited hyperparam spaces (e.g. lr between [0.1, .05] or [e-5, e-1]) for well working params --> analyze with probabilistic space
- ideas for better models:
  - u-square-net (smaller unets embedded in bigger unets)
  - unet+++ (more skip connections)
  - idea: combine both approaches, add attention unet
  - look above at "architectures"
  - problem: hyperparams of hough transform (threshold for line detection) --> there was a paper?
- research other architectures from related CV fields (e.g. object detection)
- activation and kernel initialization
- hint: use vast.ai
- maybe something similar to meanshift with bounding boxes of different sizes or similar to YOLO idea
- similar to TWERSKY/DICE loss: currently many false positives, but few false negatives --> make model learn from this information --> research different losses
- straightness
  - find relative lines with hough transform, blurr, use this as attention layer (extra hint for network)
  - perform hough transform on prediction: use info (maybe in postprocessing step to fill gaps of the same road)
  - init road with hough transform on a segment, follow the lines of the road with "brush"
  - Laplacian/Divergence operator to penalize curvature
- passwort protect mlflow
- print current epoch during training