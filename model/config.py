# albumentations?
# import albumentations as A
import torch
import os.path

from model.augmentations import transforms
from model.yolov3 import default_config, config_tiny

DATASET = os.path.join("data", "test_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
IN_CHANNELS = 1
NUM_WORKERS = 4
BATCH_SIZE = 32
BUILD_CONFIG = default_config
# TODO rename to data_size or series_size
IMAGE_SIZE = 416
NUM_CLASSES = 3
LEARNING_RATE = 1e-4
LR_EXP_DECAY_FACTOR = 0.95
WEIGHT_DECAY = 0 # 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
SCALES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
SERIES_DIR = os.path.join(DATASET, "1d_series")
LABEL_DIR = os.path.join(DATASET, "labels")
TEST_INTERVAL = 1

ANCHORS_2D = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

# these anchors have been calculated on the test_data using kmeans.py
# with 6 clusters
ANCHORS_1D = [
    [0.28, 0.38, 0.9],
    [0.07, 0.15, 0.14],
    [0.02, 0.04, 0.08],
]  # Note these have been rescaled to be between [0, 1]

# Kmeans determined optimal anchors for our dataset
ANCHORS_1D_KMEANS = [
    [0.4822033, 0.57972172],
    [0.3030703, 0.37909713],
    [0.19150963, 0.24740407],
]

scale = 1.1
train_transforms = transforms.ComposedTransform(
    transforms=[
        transforms.Translate(x_shift=transforms.UniformDistribution(-0.1, 0.1),
                             extrapolation_method="constant",
                             p=0.1
                             ),
        transforms.PolynomialBaseline(
            intercept=transforms.NormalDistribution(0, 0.1),
            linear=transforms.NormalDistribution(0, 0.01),
            quadratic=transforms.NormalDistribution(0, 0.01),
            x0=transforms.NormalDistribution(0.5, 0.05),
            p=0.5,
            ),
        transforms.Flip(
            p=0.1,
            ),
        ],
    p=0.5)
# train_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#         A.PadIfNeeded(
#             min_height=int(IMAGE_SIZE * scale),
#             min_width=int(IMAGE_SIZE * scale),
#             border_mode=cv2.BORDER_CONSTANT,
#         ),
#         A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
#         A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
#         A.OneOf(
#             [
#                 A.ShiftScaleRotate(
#                     rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
#                 ),
#                 A.IAAAffine(shear=15, p=0.5, mode="constant"),
#             ],
#             p=1.0,
#         ),
#         A.HorizontalFlip(p=0.5),
#         A.Blur(p=0.1),
#         A.CLAHE(p=0.1),
#         A.Posterize(p=0.1),
#         A.ToGray(p=0.1),
#         A.ChannelShuffle(p=0.05),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
# )
# test_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=IMAGE_SIZE),
#         A.PadIfNeeded(
#             min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
#         ),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
# )

TEST_LABELS_1D_DATA = [
    "gaussian",
    "skewed_gaussian",
    "lorentzian",
]