# DATA FORMAT
each dataset is saved as a numpy array with shape (C, N)
    + C  ==  number of channels (in images this would be RGB)
        Here it could be overlapping sets of features, like the y(x)
        values and the corresponding x values
    + N  ==  number of datapoints in one dataset (number of x_i)
        Should be a multiple of 32, 416 by default for YOLOv3

Datatype should be numpy.float32
