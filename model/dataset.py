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
                 one_anchor_per_scale=False,
                 return_bboxes=False,
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
        self._one_anchor_per_scale = one_anchor_per_scale
        self.return_bboxes = return_bboxes

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
            augmentations = self.transform(series=series, bboxes=bboxes)
            series = augmentations["series"].copy()
            bboxes = augmentations["bboxes"]

        if self.return_bboxes:
            return series, torch.tensor(bboxes)

        # Target specification
        # define a target vector [obj, x, w, class]
        # for each grid_cell
        # which has to be defined for each anchor
        # which has to be difined for each prediction scale (each grid)
        targets = [torch.zeros((self.num_anchors_per_scale, grid_size, 4))
                   for grid_size in self.grids]
        # loop over bboxes and assign cells with objects
        for bbox in bboxes:
            x, width, class_label = bbox
            # argsort the anchors by best IOU over GT width
            iou_anchors = iou_width(torch.tensor(width), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            # assign one best possible anchor for each scale
            scale_has_anchor = [False for _ in range(len(self.grids))]
            for idx in anchor_indices:
                scale_idx, anchor_on_scale_idx = self._scale_anchor_index(idx)
                grid_size = self.grids[scale_idx]
                # calculate which cell the label is defined on
                x_cell = grid_size * x
                x_cell_idx = int(x_cell)
                to_be_assigned_target = targets[scale_idx][anchor_on_scale_idx, x_cell_idx, :]
                anchor_taken = to_be_assigned_target[0]
                if not anchor_taken and not scale_has_anchor[scale_idx]:
                    # set objectness score to 0
                    x_cell_rel = x_cell - x_cell_idx
                    cell_width = width * grid_size
                    # set the other values for the given object
                    to_be_assigned_target[0] = 1
                    to_be_assigned_target[1] = x_cell_rel
                    to_be_assigned_target[2] = cell_width
                    to_be_assigned_target[3] = int(class_label)
                    if self._one_anchor_per_scale:
                        scale_has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[idx] > self.ignore_iou_thresh:
                    # ignore prediction by setting objectness to -1
                    # but only if the anchor has not yet been taken!
                    to_be_assigned_target[0] = -1
        return series, tuple(targets)

    def _scale_anchor_index(self, idx) :
        """Returns the scale_idx and anchor_on_scale_idx

        We have predefined anchors for every scale like the following
        >>> anchors = [
        >>>    [s0, s1, s2], # anchors for grid with 13 cells
        >>>    [s3, s4, s5], # anchors for grid with 26 cells
        >>>    [s6, s7, s8], # anchors for grid with 52 cells
        >>> ]

        we are given de index `idx` of a sorted list
        [idx_1, idx_2, idx_3]
        where the indices are in [0, 8]

        if the idx is 4 we want to be able to acces anchor s4
        To do this we need to return which `gridsize` s4 belongs
        to and which anchor of that list we are accessing

        To that end, we calculate the idx // num_grids
        to get the grid idx and idx % num_grids to get the
        anchor idx

        we now want to know the
        """
        scale_idx = idx // self.num_anchors_per_scale # 0, 1, 2
        anchor_on_scale_idx = idx % self.num_anchors_per_scale # 0, 1, 2
        return scale_idx, anchor_on_scale_idx


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
