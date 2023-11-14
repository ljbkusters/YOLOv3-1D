# import model.config
import numpy
import os
import pandas
import torch

from model.utils import iou_width, non_max_suppression

class YOLO1DDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_csv, series_dir, label_dir, anchors,
                 series_size=416,
                 grids=(13, 26, 52),
                 num_classes=20,
                 transform=None,
                 ignore_io_thresh = 0.5,
                 *args, **kwargs
                 ):
        """YOLO Dataset class

        Args:
            csv_file (str): file linking series to labels
            series_dir (str): directory with series data
            label_dir (str): directory with label data
            anchors (tuple[int, 3]): Number of anchors for each scale
            series_size (int, optional): series length.
                Defaults to 416. Must be multiple of 32.
            grids (tuple[int, 3], optional): grid sizes.
                Defaults to (13, 26, 52).
            num_classes (int, optional): number of identifyable classes.
                Defaults to 20.
            transform (transform, optional): Apply a transform to data
                (like shift, stretch, flip etc.). Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.annotations = pandas.read_csv(annotations_csv, comment="#",
                                           delimiter=", ")
        self.series_dir = series_dir
        self.label_dir = label_dir
        # series size might be used for the transform
        self.series_size = series_size
        self.transform = transform
        self.grids = grids
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = ignore_io_thresh

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        series_path = os.path.join(self.series_dir, self.annotations.iloc[index, 0])
        # TODO currently I'm following the img default setup where each img
        # is stored in a seperate file
        # A set of 1D data could, however, be stored in one file
        # (together with ground truth!)

        # augmentation library (seems to be specific to images though...)
        # bboxes = numpy.roll(numpy.loadtxt(fname=label_path, delimiter=" ", ndim=2), 4, axis=1).tolist()

        bboxes = numpy.loadtxt(fname=label_path, delimiter=" ",
                               comments="#", ndmin=2).tolist()
        series = numpy.load(file=series_path)

        # if data was stored as 1d data, add a dimension representing a single channel
        if series.ndim == 1:
            series = series[numpy.newaxis, :]

        if self.transform is not None:
            # self.transform:
            # takes 1D series and bboxes
            # outputs transformed
            augmetnations = self.transorm(series=series, bboxes=bboxes)
            series = augmetnations["series"]
            bboxes = augmetnations["bboxes"]

        # Target specification
        # target has size
        # (N scale predictions (3 by default),
        #  number of grid spots,
        #  (objectness_score (float), x_anchor (float), x_width (float), class (int)))
        targets = [torch.zeros((self.num_anchors_per_scale, grid_size, 4))
                   for grid_size in self.grids]

        for bbox in bboxes:
            iou_anchors = iou_width(torch.tensor(bbox[1:2]),  # width
                                    self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, width, class_label = bbox
            has_anchor = [False, False, False]

            for idx in anchor_indices:
                scale_idx = idx // self.num_anchors_per_scale # 0, 1, 2
                anchor_on_scale = idx % self.num_anchors_per_scale # 0, 1, 2
                grid_size = self.grids[scale_idx]
                # calculate which cell a label relates to
                x_cell = int(grid_size * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, x_cell, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    # set objectness score to 0
                    targets[scale_idx][anchor_on_scale, x_cell, 0] = 1
                    x_cell_rel = (x * grid_size) - x_cell
                    cell_width = width * grid_size
                    box_coordinates = torch.Tensor([x_cell_rel, cell_width])
                    # set the other values for the given object
                    targets[scale_idx][anchor_on_scale, x_cell, 1:3] = box_coordinates
                    targets[scale_idx][anchor_on_scale, x_cell, 3] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[idx] > self.ignore_iou_thresh:
                    # ignore prediction by setting objectness to -1
                    targets[scale_idx][anchor_on_scale, x_cell, 0] = -1
        return series, tuple(targets)


if __name__ == "__main__":
    # simple unittests
    series_size = 416
    num_classes = 1
    n_samples = 1
    anchors = [
            [(0.9)],
            [(0.14)],
            [(0.08)],
            ]
    dataset = YOLO1DDataset(
        annotations_csv=os.path.join("data", "unittest_data", "annotations.csv"),
        series_dir=os.path.join("data", "unittest_data", "1d_series"),
        label_dir=os.path.join("data", "unittest_data", "labels"),
        series_size=series_size,
        num_classes=num_classes,
        anchors=anchors,
        grids=(416//32, 416//16, 416//8),
    )
    series, targets = dataset[0]
    assert series.shape[1] == series_size, ("loaded series does not match"
                                            "expected datalength")
    assert tuple(targets[0].shape) == (n_samples, 416//32, 4)
    assert tuple(targets[1].shape) == (n_samples, 416//16, 4)
    assert tuple(targets[2].shape) == (n_samples, 416//8,  4)

    print(targets[0])
    print("Successfuly loaded test datapoints!")
