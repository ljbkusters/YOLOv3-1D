import torch
import tqdm
import os

import model.config
import model.dataset
import model.loss
import model.utils
import model.yolov3


def train(train_loader, yolo_model, optimizer,
          loss_function, scaler,
          scaled_anchors):
    loop = tqdm.tqdm(train_loader, leave=True)
    losses = []
    # yolo_model.to(model.config.DEVICE)
    sa0, sa1, sa2 = (
        scaled_anchors[0].to(model.config.DEVICE),
        scaled_anchors[1].to(model.config.DEVICE),
        scaled_anchors[2].to(model.config.DEVICE),
    )

    for batch_idx, (x, y) in enumerate(loop):
        # make sure all data uses CUDA or CPU device
        x = x.to(model.config.DEVICE)
        y0, y1, y2 = (
            y[0].to(model.config.DEVICE),
            y[1].to(model.config.DEVICE),
            y[2].to(model.config.DEVICE),
        )

        # cast data to correct format
        with torch.cuda.amp.autocast():
            out = yolo_model(x.half())
            loss = (loss_function(out[0].to(model.config.DEVICE), y0, sa0)
                    + loss_function(out[1].to(model.config.DEVICE), y1, sa1)
                    + loss_function(out[2].to(model.config.DEVICE), y2, sa2))
            # gradient backprop
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # progress bar update
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)


def main():
    yolov3 = model.yolov3.Yolo1DV3(
        num_classes=model.config.NUM_CLASSES,
        in_channels=model.config.IN_CHANNELS,
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
        )

    if model.config.LOAD_MODEL:
        model.utils.load_checkpoint(
            model.config.CHECKPOINT_FILE,
            model=yolov3,
            optimizer=optimizer,
            lr=model.config.LEARNING_RATE
            )

    # TODO idk why but for some reason the scaled_anhcors
    # variable has the first scale in position 0 and
    # the true scaled anchors at 1, 2, and 3
    scaled_anchors = (
        torch.tensor(model.config.ANCHORS_1D)
        * torch.tensor(model.config.SCALES)
           .unsqueeze(1)
        )

    # from model.loss __name__ == "__main__"
    # scaled_anchors = (
    #     torch.tensor(anchors)
    #     * torch.tensor(grids).unsqueeze(1)
    # )

    for epoch in range(model.config.NUM_EPOCHS):
        train(train_loader, yolov3, optimizer, loss_fn,
              scaler, scaled_anchors)
        if model.config.SAVE_MODEL:
            model.utils.save_checkpoint(yolov3,
                                        optimizer,
                                        filename=model.config.CHECKPOINT_FILE)
        if (epoch+1) % 10 == 0:
            print("On Test loader:")
            model.utils.check_class_accuracy(
                yolov3,
                test_loader,
                threshold=model.config.CONF_THRESHOLD,
            )
            pred_boxes, true_boxes = model.utils.get_evaluation_bboxes(
                test_loader, yolov3,
                iou_threshold=model.config.NMS_IOU_THRESH,
                anchors=model.config.ANCHORS_1D,
                threshold=model.config.CONF_THRESHOLD,
            )
            map_val = model.utils.mean_average_precision(
                pred_boxes, true_boxes,
                iou_threshold=model.config.NMS_IOU_THRESH,
                box_format="midpoint",
                num_classes=model.config.NUM_CLASSES,
            )
            print(f"MEAN AVERAGE PRECISION: {map_val}")


if __name__ == "__main__":
    main()
