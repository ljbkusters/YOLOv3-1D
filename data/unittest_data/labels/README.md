# DATA FORMAT
Each label csv file is assigned to one datataset in 1d_series
Each label csv file may contain multiple ground truth bounding
boxes for that series.
Each row of a label csv file contains the following entries:
    (X0, W, CLASS_LABEL)
    X0: bounding box anchor
    W: width of bounding box
    CLASS_LABEL: an integer >= 0 representing the class
