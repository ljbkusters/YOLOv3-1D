import torch
import torch.nn as nn
import numpy

from model.utils import intersection_over_union_1d

class Yolo1DLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # mean squared error loss
        self.mse = nn.MSELoss()
        # binary cross entropy with logits
        self.bce = nn.BCEWithLogitsLoss()
        # cross entropy
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # constants
        # value no object and box more than class and obj
        # these could be changed
        # even lambda_noobj = 1 should work
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):

        # indexing
        # predictions[..., 0] : objectness score
        # predictions[..., 1] : box x coordinate
        # predictions[..., 2] : box x width
        # predictions[..., 3:] : classes (one hot)

        # get objectness ground truth
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # ==============
        # No object loss
        # ==============
        # in those locations where no object is located
        # BCE loss between predicted objectness and
        # true objectness is calculated
        no_obj_loss = self.bce(
            predictions[..., 0:1][noobj],
            target[..., 0:1][noobj]
        )

        # Obj loss
        # in those cells where an object IS located,
        # BCE loss is calculated between
        # predicted objectness and IOU weighted ground_truth

        anchors = torch.reshape(anchors, (1, len(anchors), 1, 1))
        box_predictions = torch.cat([self.sigmoid(predictions[..., 0:1]),
                                     torch.exp(predictions[..., 1:2] * anchors)],
                                    dim=-1)
        # no gradients over ious
        ious = (intersection_over_union_1d(box_predictions[obj],
                                           target[..., 1:3][obj])
                .detach())
        object_loss = self.bce(predictions[..., 0:1][obj],
                               ious*target[..., 0:1][obj]
                               )

        # box coordinate loss
        # modify target instead of predictions for better gradient flow
        # (because log is better behaved than exp)
        # this step trains BBOX prediction
        predictions[..., 1:2] = self.sigmoid(predictions[..., 1:2]) #x between 0, 1
        target[..., 2:3] = torch.log(
            target[..., 2:3] / anchors + 1e-6
        ) # width
        box_loss = self.mse(predictions[..., 1:3][obj],
                            target[..., 1:3][obj])

        # class loss
        # this step trains class prediction
        class_loss = self.entropy(
            (predictions[..., 3:][obj]),
            (target[..., 3][obj].long()
             # by casting this float to long()
             # the cross entropy function
             # automatically handles integer
             # to ONE HOT conversion
             # (this does not happen with floats!)
             )
        )
        return (
            self.lambda_box * box_loss
            + self.lambda_class * class_loss
            + self.lambda_noobj * no_obj_loss
            + self.lambda_obj * object_loss
        )

if __name__ == "__main__":
    # simple unittests
    import model.dataset
    import model.yolov3
    import os.path
    yolo_loss = Yolo1DLoss()
    series_size = 416
    num_classes = 1
    n_samples = 1
    anchors = [
            [(0.9)],
            [(0.14)],
            [(0.08)],
            ]
    dataset = model.dataset.YOLO1DDataset(
        annotations_csv=os.path.join("data", "annotations.csv"),
        series_dir=os.path.join("data", "1d_series"),
        label_dir=os.path.join("data", "labels"),
        series_size=series_size,
        num_classes=num_classes,
        anchors=anchors,
        grids=(416//32, 416//16, 416//8),
    )
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=1)
    data_iter = iter(dl)
    series, targets = next(data_iter)
    print(type(series))

    model = model.yolov3.Yolo1DV3(
        in_channels=1,
        num_classes=1,
        )
    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor((13, 26, 52)).unsqueeze(1)
    )
    print(scaled_anchors)
    prediction = model(series)
    loss = Yolo1DLoss()
    # for every scale
    with torch.cuda.amp.autocast():
        for p, t, a in zip(prediction, targets, scaled_anchors):
            loss(p, t, a)
    print("succesfully calculated loss!")