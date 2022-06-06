#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys,os, gzip
import numpy as np
import _pickle as cPickle

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data_mnist(path='../cross_sim/data/datasets/mnist/mnist.npz',\
    nstart=None,nend=None,calibration=False,training=False):
    """Loads the MNIST dataset.
    # If calibration is True, x_test will be a random subset of x_train
    """
    if not training:
        if not calibration:
            with np.load(path, allow_pickle=True) as f:
                x_test, y_test = f['x_test'], f['y_test']
            x_test = x_test.reshape((10000, 28, 28, 1))
            if nstart is not None and nend is not None:
                x_test = x_test[nstart:nend,:,:,:]
                y_test = y_test[nstart:nend]

        else:
            with np.load(path, allow_pickle=True) as f:
                x_train, y_train = f['x_train'], f['y_train']
            ntest = nend - nstart
            x_train = x_train.reshape((len(y_train), 28, 28, 1))
            rand_order = np.arange(len(y_train))
            np.random.shuffle(rand_order)
            x_test = np.zeros((ntest, 28, 28, 1))
            y_test = np.zeros(ntest)
            for k in range(ntest):
                x_test[k,:,:,:] = x_train[rand_order[k],:,:,:]
                y_test[k] = y_train[rand_order[k]]

        return (x_test, y_test)

    else:
        with np.load(path, allow_pickle=True) as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
            x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
            x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
        return (x_train, y_train), (x_test, y_test)


def load_data_fashion_mnist(path='../cross_sim/data/datasets/fashion_mnist/',\
    nstart=None,nend=None,calibration=False,training=False):
    """Loads the Fashion-MNIST dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(path+fname)

    if not training:
        if not calibration:
            with gzip.open(paths[2], 'rb') as lbpath:
                y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open(paths[3], 'rb') as imgpath:
                x_test = np.frombuffer(imgpath.read(), np.uint8,
                                       offset=16).reshape(len(y_test), 28, 28, 1)
            if nstart is not None and nend is not None:
                x_test = x_test[nstart:nend,:,:,:]
                y_test = y_test[nstart:nend]

        else:
            with gzip.open(paths[0], 'rb') as lbpath:
                y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open(paths[1], 'rb') as imgpath:
                x_train = np.frombuffer(imgpath.read(), np.uint8,
                                        offset=16).reshape(len(y_train), 28, 28, 1)
            ntest = nend - nstart
            rand_order = np.arange(len(y_train))
            np.random.shuffle(rand_order)
            x_test = np.zeros((ntest, 28, 28, 1))
            y_test = np.zeros(ntest)
            for k in range(ntest):
                x_test[k,:,:,:] = x_train[rand_order[k],:,:,:]
                y_test[k] = y_train[rand_order[k]]

        return (x_test, y_test)

    else:
        with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[1], 'rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(), np.uint8,offset=16)
            x_train = x_train.reshape((len(y_train),int(len(x_train)/len(y_train))))
        with gzip.open(paths[2], 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[3], 'rb') as imgpath:
            x_test = np.frombuffer(imgpath.read(), np.uint8,offset=16)
            x_test = x_test.reshape((len(y_test),int(len(x_test)/len(y_test))))

        return (x_train, y_train), (x_test, y_test)


def load_cifar_10(path='../cross_sim/data/datasets/cifar10/',\
    nstart=None,nend=None,calibration=False,subtract_pixel_mean=False):

    if not calibration:
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)
        y_test = np.array(y_test)
        
        # Put channel last
        x_test = x_test.transpose(0, 2, 3, 1).astype(np.float32)
        if nstart is not None and nend is not None:
            x_test = x_test[nstart:nend,:,:,:]
            y_test = y_test[nstart:nend]

        if subtract_pixel_mean:
            x_train_mean = np.load(os.path.join(path, 'x_train_mean.npy'))
            x_test -= x_train_mean

    else:
        x_train = np.empty((50000, 3, 32, 32), dtype='uint8')
        y_train = np.empty((50000,), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000: i * 10000, :, :, :],
             y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)
        y_train = np.array(y_train)
        x_train = x_train.transpose(0, 2, 3, 1).astype(np.float32)

        ntest = nend - nstart
        rand_order = np.arange(len(y_train))
        np.random.shuffle(rand_order)
        x_test = np.zeros((ntest, 32, 32, 3))
        y_test = np.zeros(ntest)
        for k in range(ntest):
            x_test[k,:,:,:] = x_train[rand_order[k],:,:,:]
            y_test[k] = y_train[rand_order[k]]

        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_test -= x_train_mean

    return (x_test, y_test)


def load_cifar_100(path='../cross_sim/data/datasets/cifar100/',\
    nstart=None,nend=None,calibration=False,subtract_pixel_mean=False):

    if not calibration:
        fpath = os.path.join(path, 'test')
        x_test, y_test = load_batch(fpath, label_key='fine_labels')
        y_test = np.array(y_test)
        x_test = x_test.transpose(0, 2, 3, 1).astype(np.float32)
        if nstart is not None and nend is not None:
            x_test = x_test[nstart:nend,:,:,:]
            y_test = y_test[nstart:nend]

        if subtract_pixel_mean:
            x_train_mean = np.load(os.path.join(path, 'x_train_mean.npy'))
            x_test -= x_train_mean

    else:
        x_train[:25000,:,:,:] = np.load(os.path.join(path, 'x_train_pt1.npy'))
        x_train[25000:,:,:,:] = np.load(os.path.join(path, 'x_train_pt2.npy'))
        y_train = np.load(os.path.join(path, 'y_train.npy'))
        x_train = x_train.transpose(0, 2, 3, 1).astype(np.float32)

        ntest = nend - nstart
        rand_order = np.arange(len(y_train))
        np.random.shuffle(rand_order)
        x_test = np.zeros((ntest, 32, 32, 3))
        y_test = np.zeros(ntest)
        for k in range(ntest):
            x_test[k,:,:,:] = x_train[rand_order[k],:,:,:]
            y_test[k] = y_train[rand_order[k]]

        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_test -= x_train_mean

    return (x_test, y_test)


def load_imagenet(option, calibration=False, nstart=0,nend=9999):

    # Enter the path to the ImageNet **preprocessed** validation set images here
    path = '../cross_sim/data/datasets/imagenet/'

    if not os.path.isdir(path):
        raise ValueError("Pre-processed ImageNet images not found in directory specified in load_imagenet()")

    # Three options for preprocessed ImageNet inputs
    # "keras" : use Keras preprocessing function, up to 20k
    # "mlperf_XX" : use cv2 preprocessing function in MLPerf official benchmark, XX=ResNet50, MobileNet
    # "torchvision" : use torchvision preprocessing function in NVIDIA's int4 network (MLPerf Open division)

    # Extra offset from the start of the npy file (default zero)
    offset = 0
    ntest = nend - nstart

    # Currently the max number of images in an .npy file is 25000
    # If a larger file is possible on your machine, delete the four lines below and modify the logic in the rest of the file accordingly
    if ntest > 25000:
        raise ValueError("Number of requested images exceeds .npy size")
    if nstart < 25000 and nend > 25000:
        raise ValueError("Requested image batch spans two max size .npy files")

    # Labels
    if calibration:
        print("Loading MLPerf calibration subset of ImageNet (Keras preprocessing)")
        if ntest > 500 or nend > 500:
            raise ValueError("MLPerf calibration set only has 500 images")
        y_test = np.load(path + "y_val_MLperf_cal.npy")
    else:
        y_test = np.load(path +"y_val.npy")
    y_test = y_test[nstart:nend]

    ####
    ####    Keras models / Keras preprocessing (CV2)
    ####
    if option == "keras_resnet" or option == "keras_vgg" or option == "keras_mobilenet" or option == "keras_inception":
        # Load second step pre-processing functions (applied to cropped images)
        if option == "keras_resnet":
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif option == "keras_vgg":
            from tensorflow.keras.applications.vgg19 import preprocess_input
        elif option == "keras_mobilenet":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif option == "keras_inception":
            from tensorflow.keras.applications.inception_v3 import preprocess_input

        if calibration:
            if option != "keras_inception":
                x_test = np.load(path + "x_val_keras_cal.npy")
            else:
                x_test = np.load(path + "x_val_inception_cal.npy")

        elif option != "keras_inception":
            if nend <= 50:
                x_test = np.load(path + "x_val_keras_50.npy")
            elif nend <= 1000:
                x_test = np.load(path + "x_val_keras_1000.npy")
            elif nend <= 5000:
                x_test = np.load(path + "x_val_keras_5000.npy")
            elif nstart < 25000 and ntest <= 25000:
                x_test = np.load(path + "x_val_keras_25000_front.npy")
            elif nstart >= 25000 and ntest <= 25000:
                x_test = np.load(path + "x_val_keras_25000_back.npy")
                offset = 25000
        else:
            if nend <= 50:
                x_test = np.load(path + "x_val_inception_50.npy")
            elif nend <= 1000:
                x_test = np.load(path + "x_val_inception_1000.npy")
            elif nend <= 5000:
                x_test = np.load(path + "x_val_inception_5000.npy")
            elif nstart < 25000 and ntest <= 25000:
                x_test = np.load(path + "x_val_inception_25000_front.npy")
            elif nstart >= 25000 and ntest <= 25000:
                x_test = np.load(path + "x_val_inception_25000_back.npy")
                offset = 25000
        
        x_test = x_test[(nstart-offset):(nend-offset),:,:,:]
        x_test = preprocess_input(x_test) # converted to BGR
        return (x_test, y_test)

    ####
    ####    Torchvision preprocessing
    ####
    elif option == "torchvision":
        if calibration:
            x_test = np.load(path + "x_val_torch_cal.npy")
        elif nend <= 50:
            x_test = np.load(path + "x_val_torch_50.npy")
        elif nend <= 1000:
            x_test = np.load(path + "x_val_torch_1000.npy")
        elif nend <= 5000:
            x_test = np.load(path + "x_val_torch_5000.npy")
        elif nstart < 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_torch_25000_front.npy")
        elif nstart >= 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_torch_25000_back.npy")
            offset = 25000
        x_test = x_test[(nstart-offset):(nend-offset),:,:,:]
        return (x_test, y_test)

    ####
    ####    MLPerf preprocessing (CV2)
    ####
    elif option == "mlperf_resnet50":
        # These models have an offset of one in the class labels
        y_test += 1
        if calibration:
            x_test = np.load(path + "x_val_MLperfRN50_cal.npy")
        elif nend <= 50:
            x_test = np.load(path + "x_val_MLperfRN50_50.npy")
        elif nend <= 1000:
            x_test = np.load(path + "x_val_MLperfRN50_1000.npy")
        elif nend <= 5000:
            x_test = np.load(path + "x_val_MLperfRN50_5000.npy")
        elif nstart < 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_MLperfRN50_25000_front.npy")
        elif nstart >= 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_MLperfRN50_25000_back.npy")
            offset = 25000
        x_test = x_test[(nstart-offset):(nend-offset),:,:,:]
        return (x_test, y_test)

    elif option == "mlperf_mobilenet":
        # These models have an offset of one in the class labels
        y_test += 1
        if calibration:
            x_test = np.load(path + "x_val_MLperfMobileNet_cal.npy")
        elif nend <= 50:
            x_test = np.load(path + "x_val_MLperfMobileNet_50.npy")
        elif nend <= 1000:
            x_test = np.load(path + "x_val_MLperfMobileNet_1000.npy")
        elif nend <= 5000:
            x_test = np.load(path + "x_val_MLperfMobileNet_5000.npy")
        elif nstart < 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_MLperfMobileNet_25000_front.npy")
        elif nstart >= 25000 and ntest <= 25000:
            x_test = np.load(path + "x_val_MLperfMobileNet_25000_back.npy")
            offset = 25000
        x_test = x_test[(nstart-offset):(nend-offset),:,:,:]
        return (x_test, y_test)
