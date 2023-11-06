import os
import numpy
import matplotlib.pyplot as plt

import torch
import tqdm

import model.config
import model.yolov3
import model.loss
import model.utils

import argparse


def make_predictions(yolo_model, data_loader):

    loop = tqdm.tqdm(data_loader, leave=True)
    bbox_predictions = []
    # yolo_model.to(model.config.DEVICE)
    for batch_idx, (x, y) in enumerate(loop):
        # make sure all data uses CUDA or CPU device
        x = x.to(model.config.DEVICE)
        # y0, y1, y2 = (
        #     y[0].to(model.config.DEVICE),
        #     y[1].to(model.config.DEVICE),
        #     y[2].to(model.config.DEVICE),
        # )

        # cast data to correct format
        with torch.cuda.amp.autocast():
            out = yolo_model(x.half())
            bboxes = [model.utils.cells_to_bboxes(
                scale_prediction,
                model.config.ANCHORS_1D,
                model.config.SCALES,
                ) for scale_prediction in out]
            bbox_predictions.append(bboxes)
    return bbox_predictions

def visualize_data(loader):
    import matplotlib.pyplot as plt
    for batch_idx, (x_batch, (y0b, y1b, y2b)) in enumerate(loader):
        for x, y0, y1, y2 in zip(x_batch, y0b, y1b, y2b):
            X = x[0, :]
            x_linspace = numpy.linspace(0, 1, len(X))
            plt.plot(x_linspace, X)
            # y0 13 cells
            for i in range(13):
                objectness, cell_mean, cell_width, class_lbl = y0[0, i].numpy()
                if objectness > 0.5:
                    print("cell mu, cell width")
                    print(cell_mean, cell_width)
                    mu = (i+cell_mean)/13
                    w = cell_width/13
                    x0 = mu - w/2
                    x1 = mu + w/2
                    print("curve mu, curve width")
                    print(mu, w)
                    print("x0, x1")
                    print(x0, x1)
                    plt.axvspan(x0, x1, color="green", alpha=0.5)
                    plt.text(mu, 0, class_lbl)
            plt.show()
            # y1 26 cells
            # y2 52 cells

def main(args):
    yolov3 = model.yolov3.Yolo1DV3(
        num_classes=model.config.NUM_CLASSES,
        in_channels=model.config.IN_CHANNELS,
        num_anchors_per_scale=args.aps,
        ).to(model.config.DEVICE)
    optimizer = torch.optim.Adam(
        yolov3.parameters(),
        lr=model.config.LEARNING_RATE,
        weight_decay=model.config.WEIGHT_DECAY
        )

    loss_fn = model.loss.Yolo1DLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = \
        model.utils.get_loaders(
            train_csv_path=os.path.join(
                model.config.DATASET, "train_annotations.csv"),
            test_csv_path=os.path.join(
                model.config.DATASET, "test_annotations.csv"),
            shuffle=False,
        )

    model.utils.load_checkpoint(
        args.checkpoint_file,
        model=yolov3,
        optimizer=optimizer,
        lr=model.config.LEARNING_RATE
        )

    print("=> making predictions on test data")
    pred_boxes, true_boxes = model.utils.get_evaluation_bboxes(
        test_loader, yolov3,
        iou_threshold=model.config.NMS_IOU_THRESH,
        anchors=model.config.ANCHORS_1D,
        threshold=model.config.CONF_THRESHOLD,
        box_format="midpoint"
    )
    print(len(pred_boxes))
    print(len(true_boxes))
    print(len(true_boxes[0]))
    print(true_boxes[0])
    print(true_boxes[1])

    # load curves
    curves = []
    for i, (x_batch, y_batch) in enumerate(test_loader):
        #print(type(x_batch))
        x_batch = x_batch.numpy()
        #print(type(x_batch))
        #print(x_batch)
        curves.extend([x_batch[i, 0, :] for i in range(x_batch.shape[0])])

    # yikes very inefficient code below
    # but... quick and dirty solution

    # plot curves with bboxes
    x_linspace = numpy.linspace(0, 1, 416)
    for i, curve in enumerate(curves):
        # get all true bboxes
        pred_labels = []
        plt.figure(figsize=(6, 6))
        for bbox in true_boxes:
            if bbox[0] == i:
                idx, clslbl, conf, xm, w = bbox
                str_lbl = model.config.TEST_LABELS_1D_DATA[int(clslbl)]
                x0 = xm - w/2
                x1 = xm + w/2
                plt.axvspan(x0, x1, color='green', alpha=0.2)
                plt.axvline(x0, color='green')
                plt.axvline(x1, color='green')
                lbl = f"true_class: {str_lbl},\n $\mu:$ {xm:.3f}, $w$: {w:.3f}"
                plt.text(0.1, 0.9, lbl, color="green",
                         transform=plt.gca().transAxes,
                         bbox=dict(facecolor='white',
                                   alpha=0.7,
                                   edgecolor='green',
                                   boxstyle='round')
                         )
        # get all predicted bboxes
        for bbox in pred_boxes:
            if bbox[0] == i:
                idx, clslbl, conf, xm, w = bbox
                str_lbl = model.config.TEST_LABELS_1D_DATA[int(clslbl)]
                x0 = xm - w/2
                x1 = xm + w/2
                plt.axvspan(x0, x1, color='red', alpha=0.5)
                lbl = (f"class: {str_lbl}\nconf: {conf:.3f}\n"
                       f"$\mu$ {xm:.3f}, $w$: {w:.3f}")
                pred_labels.append(lbl)
        plt.text(0.1, 0.7, "\n".join(pred_labels), color="red",
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white',
                           alpha=0.7,
                           edgecolor='red',
                           boxstyle='round')
                 )
        plt.xlabel("normalized domain")
        plt.ylabel("normalized height")
        plt.plot(x_linspace, curve)
        plt.show()

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", help="checkpoint loading file")
    parser.add_argument("--aps", help="anchors per scale", type=int)
    args = parser.parse_args()

    # main
    main(args)
