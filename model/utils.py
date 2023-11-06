"""utils.py

Utility functions for training and evaluation

TODO rewrite functions for 1D data
"""
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import config


def iou_width(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width of the first bounding boxes
        boxes2 (tensor): width of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1, boxes2)
    union = boxes1 + boxes2 - intersection
    return intersection / union


def intersection_over_union_1d(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 1:2] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 1:2] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 1:2] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 1:2] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_x2 = boxes_preds[..., 1:2]
        box2_x1 = boxes_labels[..., 0:1]
        box2_x2 = boxes_labels[..., 1:2]

    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)

    intersection = (x2 - x1).clamp(0)
    box1_length = abs((box1_x2 - box1_x1))
    box2_length = abs((box2_x2 - box2_x1))

    return intersection / (box1_length + box2_length - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, x2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union_1d(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20
    ):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union_1d(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_curve(curve, boxes):
    """Plots predicted bounding boxes on the image"""
    class_labels = config.TEST_LABELS_1D_DATA

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.plot(curve.x, curve.y)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

    # Create bounding domains by axvlines patch
    for box in boxes:
        assert len(box) == 4, "box should contain class pred, confidence, x, width"
        class_label, objectness, x0, x1 = box
        cl = int(class_label)
        onset = x0 - x1/2
        end = x0 + x1/2
        ax.axvline(onset, color=colors[cl])
        ax.axvline(end, color=colors[cl])
        ax.axvspan(onset, end, color=colors[cl], alpha=0.5)
    plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
    ):
    """Get prediction and ground truth bboxes

    This is mainly usefull for visualization purposes

    Args:
        loader (torch.Dataloader | Iterable): A Dataloader object
            returning (x, y)
        model (torch.Model): PyTorch model
        iou_threshold (float): NMS rejection threshold, overlapping
            bboxes above the iou threshold are assumed to predict
            the same object and the bbox with the lowest confidence
            is removed.
        anchors (Iterable): Iterable (array-like) of base anchors
        threshold (float): Confidence threshold.
        box_format (str, optional): Input format of bboxes to non-max
            suppression.  Either "corners" or "midpoint".
            Defaults to "midpoint".
        device (str, optional): PyTorch device, either "cpu", a gpu
        device or "cuda". Defaults to "cuda".

    Returns:
        all_pred_bboxes, all_true_bboxes[list]: lists of bboxes
            (predicted and ground truth).  bbox specification:
            [idx, class, confidence, x0, width]
            here idx refers to the i-th evaluated input
    """
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x.float())

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        # TODO use len(scales)
        for scale_idx in range(3):
            S = predictions[scale_idx].shape[2]
            anchor = torch.tensor([*anchors[scale_idx]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[scale_idx], anchor, S=S, is_preds=True
            )
            for batch_idx, (box) in enumerate(boxes_scale_i):
                bboxes[batch_idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 1, S, 3+num_classes)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, 1+3) with class index,
                      object score, bounding box coordinates

    prediction to bbox
    1. calculate variables relative to cell
    prediction (N, n_cells @ scale, n_anchors @ scale, (obj_score, t_x0, t_w, *one_hot_casses))
    cell_x0 = sigmoid(t_x0) (relative to cell)
    cell_w = base_anchor * exp(t_w) (relative to cell)
    best class = argmax(one_hot_classes)

    1. calculate variables relative to input
    x0 = (cell_idx + x0) * cell scale (relative to input)
    w = (width) * cell scale (relative to input)
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:3]
    if is_preds:
        # reshape anchors to correct shape
        anchors = anchors.reshape(1, len(anchors), 1, 1)
        # print(anchors.shape)
        # convert x0 prediction
        box_predictions[..., 0:1] = torch.sigmoid(box_predictions[..., 0:1])
        # convert width prediction
        box_predictions[..., 1:2] = torch.exp(box_predictions[..., 1:2]) * anchors
        # convert objectness score
        scores = torch.sigmoid(predictions[..., 0:1])
        # convert one_hot class predictions to single best class
        best_class = torch.argmax(predictions[..., 3:], dim=-1).unsqueeze(-1)
    else:
        # simply take the score
        scores = predictions[..., 0:1]
        # simply take the class prediciton
        best_class = predictions[..., 3:4]

    # step 2 calculate bbox relative to input instead of relative to cell
    # this should be adding the cell index
    cell_indices = (
        torch.arange(S)
        .repeat(BATCH_SIZE, 1, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    # print(cell_indices.shape)
    # rescale to input
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    w = 1 / S * box_predictions[..., 1:2]
    # print(best_class.shape, scores.shape, x.shape, w.shape)
    # we now `squeeze' the predictions per anchor and per cell into one dimension
    # and get n_anchors * S predictions per batch_index
    converted_bboxes = torch.cat((best_class, scores, x, w), dim=-1).reshape(BATCH_SIZE, num_anchors * S, 4)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x.float())

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 3:][obj], dim=-1) == y[i][..., 3][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path, shuffle=True):
    from model.dataset import YOLO1DDataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLO1DDataset(
        train_csv_path,
        #transform=config.train_transforms,
        grids=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        series_dir=config.SERIES_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS_1D,
    )
    test_dataset = YOLO1DDataset(
        test_csv_path,
        # transform=config.test_transforms,
        grids=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        series_dir=config.SERIES_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS_1D,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=False,
    )

    train_eval_dataset = YOLO1DDataset(
        train_csv_path,
        # transform=config.test_transforms,
        grids=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        series_dir=config.SERIES_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS_1D,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_curve(x[i].permute(1,2,0).detach().cpu(), nms_boxes)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # simple unittests
    # seed everything such that torch.rand is the same (also tests this function)
    seed_everything()

    # test cells_to_bboxes
    BATCH_DIM = 64
    ANCHORS = torch.tensor([(0.9, 0.8), (0.14, 0.10), (0.08, 0.03)])
    CELLS = [416//32, 416//16, 416//8]
    n_classes = 4
    predictions = torch.rand(BATCH_DIM, len(ANCHORS[0]), CELLS[0], 3+n_classes)
    # print(predictions.shape)
    bboxes = cells_to_bboxes(predictions, ANCHORS[0], CELLS[0], is_preds=True)
    # bboxes are specified (batch_dim, cells, (class, objectness, x0, width))
    print(len(bboxes[0]))
    assert (len(bboxes) == BATCH_DIM), ("prediction has wrong number of cells "
                                        f"(should be {BATCH_DIM})")
    assert (len(bboxes[0]) == len(ANCHORS[0]) * CELLS[0]), ("prediction has wrong number of cells "
                                         f"(should be {len(ANCHORS[0]) * CELLS[0]})")
    assert (len(bboxes[0][0]) == 4), ("prediction has wrong dimension (should be 4)")
    for i in range(BATCH_DIM):
        for j in range(len(ANCHORS[0]) * CELLS[0]):
            assert (int(bboxes[i][j][0]) < n_classes), ("found an impossible class prediction")

    # test plot_curve
    import collections, numpy
    Curve = collections.namedtuple("Curve", "x y")
    x = numpy.linspace(0, 1, 100)
    y = 0.5 * x
    curve = Curve(x, y)
    print("plotting example curve with nonsensical data and random classes")
    print("this should show a line with 3 class predictions as colored regions")
    plot_curve(curve, [bboxes[0][i] for i in numpy.random.randint(low=0, high=25, size=3)])
    user_input = input("does the plot look as expected? (y/n): ")
    while user_input not in ("y", "yes", "n", "no"):
        print("please answer with 'y', 'n', 'yes', or 'no' ")
        user_input = input("does the plot look as expected? (y/n): ")
    if user_input[0] == "y":
        print("succesfully plotted data")
    else:
        raise RuntimeWarning("User rejected plot")

    # test


def keyboard_interruptable(func):
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except KeyboardInterrupt:
            print("EARLY EXIT DUE TO KEYBOARD INTERRUPT!")
            exit(0)
    return wrapper