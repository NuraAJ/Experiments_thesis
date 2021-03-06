#try:
#    import sys
#    import os
#    import pickle as pkl
#    import time
#    import scipy
#    #import matplotlib.pyplot as plt
#    import scipy.misc
#    import numpy as np
#    #from scipy.io import loadmat
#    import tensorflow as tf
#    import argparse
#    from time import gmtime, strftime
#except ImportError as e:
#    print(e)


import sys
import os
import pickle as pkl
import time
import scipy
import scipy.misc
import numpy as np
import tensorflow as tf
import argparse
from time import gmtime, strftime
np.random.seed(42)

#Load the images from the given directory
#Images should be saved like:
#Parent Dir-
#-----------Class 1
#----------------Image_1.jpg
#----------------Image_2.jpg
#----------------Image_3.jpg
#----------------....
#----------------Image_m.jpg
#-----------Class 2
#-----------Class 3
# . . .
#-----------Class n
def load_create_data(DATA_DIR):
    global extra_class
    extra_class = 0
    DATA_DIR = DATA_DIR
    classes = os.listdir(DATA_DIR)

    all_data = []
    y_data = []

    for id_class in range(len(classes)):
        each_class = classes[id_class]
        if each_class == '.DS_Store':
            continue
        files = os.listdir(DATA_DIR + '/' + each_class)

        for im_path in files:
            full_path = DATA_DIR + '/' + each_class + '/' + im_path
            if im_path == '.DS_Store':
                continue
            im = scipy.misc.imread(full_path)
            im = scipy.misc.imresize(im, (32, 32))
            all_data.append(im)
            y_data.append(id_class)
            if not im.shape == (32, 32, 3):
                print(each_class + '/' + im_path)

    X_data = np.zeros((32, 32, 3, len(y_data)), dtype=np.uint8)
    for ix in range(len(y_data)):
        X_data[:, :, :, ix] = all_data[ix]
    y_data = np.asarray(y_data)
    global trainset, testset
    trainset = {'X': None, 'y': None}
    testset = {'X': None, 'y': None}
    # Decide what % of the data goes into training
    split = int(0.90 * len(y_data))
    idx = range(len(y_data))
    idx = list(idx)
    np.random.shuffle(idx)

    trainset['X'] = X_data[:, :, :, idx[:split]]
    testset['X'] = X_data[:, :, :, idx[split:]]

    trainset['y'] = y_data[idx[:split]]
    testset['y'] = y_data[idx[split:]]


############################################################
def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x
############################################################
##### Activation Function #####
def leaky_relu(x, alpha=0.2, name=None):
    return tf.maximum(alpha * x, x)

################ CREATING THE DATASET #####################
class Dataset:
    def __init__(self, train, test, val_frac=0.0, shuffle=True, scale_func=None):
        split_idx = int(len(test['y']) * (1 - val_frac))
        self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[0:len(self.train_y)] = 1
        print(len(self.test_y))
        self.train_x = np.rollaxis(self.train_x, 3)
        #self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler(self.train_x)
        #self.valid_x = self.scaler(self.valid_x)
        self.test_x = self.scaler(self.test_x)
        self.shuffle = shuffle

    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        num_examples = len(getattr(dataset, y_name))
        if self.shuffle:
            idx = np.arange(num_examples)
            np.random.shuffle(idx)
            setattr(dataset, x_name, getattr(dataset, x_name)[idx])
            setattr(dataset, y_name, getattr(dataset, y_name)[idx])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[idx]

        dataset_x = getattr(dataset, x_name)
        dataset_y = getattr(dataset, y_name)
        for ii in range(0, num_examples, batch_size):
            x = dataset_x[ii:ii + batch_size]
            y = dataset_y[ii:ii + batch_size]

            if which_set == "train":
                yield x, y, self.label_mask[ii:ii + batch_size]
            else:
                yield x, y

#####################################

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    y = tf.placeholder(tf.int32, (None), name='y')
    label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

    return inputs_real, inputs_z, y, label_mask


#####################################
def discriminator(x, reuse=False, alpha=0.2, drop_rate=0.5, num_classes=10, size_mult=64):
    print("\n###################################################\n")
    print("\nDiscriminator Architecture")
    result_file.write("\n###################################################\n")
    result_file.write("\nDiscriminator Architecture")
    
    with tf.variable_scope('discriminator', reuse=reuse):
        print("\nInput Image")
        result_file.write("\nInput Image")
        print("\nconv2d(Input, filters, kernel_size, strides, padding)")
        result_file.write("\nconv2d(Input, filters, kernel_size, strides, padding)")
        
        print("\n\nconv2d(x, 64, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
              
        result_file.write("\n\nconv2d(x, 64, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_1 = tf.layers.conv2d(x, 64, [3, 3], [1, 1], padding="same")
        lrelu_1 = leaky_relu(x_1, alpha)
        bn_1 = tf.layers.batch_normalization(lrelu_1, training=True)
        drop_1 = tf.layers.dropout(bn_1, rate=drop_rate)


        print("\n\nconv2d(x, 64, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_1, 64, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_2 = tf.layers.conv2d(drop_1, 64, [3, 3], [1, 1], padding="same")
        lrelu_2 = leaky_relu(x_2, alpha)
        bn_2 = tf.layers.batch_normalization(lrelu_2, training=True)
        drop_2 = tf.layers.dropout(bn_2, rate=drop_rate)


        print("\n\nmax_pooling2d(drop_2, [ 2, 2], [2, 2], padding='same')")
        result_file.write("\n\nmax_pooling2d(drop_2, [ 2, 2], [2, 2], padding='same')")
        pool_1 = tf.layers.max_pooling2d(drop_2, [ 2, 2], [2, 2], padding="same")


        print("\n\nconv2d(pool_1, 128, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(pool_1, 128, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_3 = tf.layers.conv2d(pool_1, 128, [3, 3], [1, 1], padding="same")
        lrelu_3 = leaky_relu(x_3, alpha)
        bn_3 = tf.layers.batch_normalization(lrelu_3, training=True)
        drop_3 = tf.layers.dropout(bn_3, rate=drop_rate)


        print("\n\nconv2d(drop_3, 128, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_3, 128, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_4 = tf.layers.conv2d(drop_3, 128, [3, 3], [1, 1], padding="same")
        lrelu_4 = leaky_relu(x_4, alpha)
        bn_4 = tf.layers.batch_normalization(lrelu_4, training=True)
        drop_4 = tf.layers.dropout(bn_4, rate=drop_rate)

        print("\n\nmax_pooling2d(drop_4, [ 2, 2], [2, 2], padding='same')")
        result_file.write("\n\nmax_pooling2d(drop_4, [ 2, 2], [2, 2], padding='same')")
        pool_2 = tf.layers.max_pooling2d(drop_4, [2, 2], [2, 2], padding="same")


        print("\n\nconv2d(pool_2, 256, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(pool_2, 256, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_5 = tf.layers.conv2d(pool_2, 256, [3, 3], [1, 1], padding="same")
        lrelu_5 = leaky_relu(x_5, alpha)
        bn_5 = tf.layers.batch_normalization(lrelu_5, training=True)
        drop_5 = tf.layers.dropout(bn_5, rate=drop_rate)



        print("\n\nconv2d(drop_5, 256, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_5, 256, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_6 = tf.layers.conv2d(drop_5, 256, [3, 3], [1, 1], padding="same")
        lrelu_6 = leaky_relu(x_6, alpha)
        bn_6 = tf.layers.batch_normalization(lrelu_6, training=True)
        drop_6 = tf.layers.dropout(bn_6, rate=drop_rate)


        print("\n\nconv2d(drop_6, 256, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_6, 256, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_7 = tf.layers.conv2d(drop_6, 256, [3, 3], [1, 1], padding="same")
        lrelu_7 = leaky_relu(x_7, alpha)
        bn_7 = tf.layers.batch_normalization(lrelu_7, training=True)
        drop_7 = tf.layers.dropout(bn_7, rate=drop_rate)



        print("\n\nconv2d(drop_7, 256, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_7, 256, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_8 = tf.layers.conv2d(drop_7, 256, [3, 3], [1, 1], padding="same")
        lrelu_8 = leaky_relu(x_8, alpha)
        bn_8 = tf.layers.batch_normalization(lrelu_8, training=True)
        drop_8 = tf.layers.dropout(bn_8, rate=drop_rate)

        print("\n\nmax_pooling2d(drop_8, [ 2, 2], [2, 2], padding='same')")
        result_file.write("\n\nmax_pooling2d(drop_8, [ 2, 2], [2, 2], padding='same')")
        pool_3 = tf.layers.max_pooling2d(drop_8, [ 2, 2], [2, 2], padding="same")


        print("\n\nconv2d(pool_3, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(pool_3, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_9 = tf.layers.conv2d(pool_3, 512, [3, 3], [1, 1], padding="same")
        lrelu_9 = leaky_relu(x_9, alpha)
        bn_9 = tf.layers.batch_normalization(lrelu_9, training=True)
        drop_9 = tf.layers.dropout(bn_9, rate=drop_rate)



        print("\n\nconv2d(drop_9, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_9, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_10 = tf.layers.conv2d(drop_9, 512, [3, 3], [1, 1], padding="same")
        lrelu_10 = leaky_relu(x_10, alpha)
        bn_10 = tf.layers.batch_normalization(lrelu_10, training=True)
        drop_10 = tf.layers.dropout(bn_10, rate=drop_rate)


        print("\n\nconv2d(drop_10, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_10, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_11 = tf.layers.conv2d(drop_10, 512, [3, 3], [1, 1], padding="same")
        lrelu_11 = leaky_relu(x_11, alpha)
        bn_11 = tf.layers.batch_normalization(lrelu_11, training=True)
        drop_11 = tf.layers.dropout(bn_11, rate=drop_rate)


        print("\n\nconv2d(drop_11, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_11, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_12 = tf.layers.conv2d(drop_11, 512, [3, 3], [1, 1], padding="same")
        lrelu_12 = leaky_relu(x_12, alpha)
        bn_12 = tf.layers.batch_normalization(lrelu_12, training=True)
        drop_12 = tf.layers.dropout(bn_12, rate=drop_rate)

        print("\n\nmax_pooling2d(drop_12, [ 2, 2], [2, 2], padding='same')")
        result_file.write("\n\nmax_pooling2d(drop_12, [ 2, 2], [2, 2], padding='same')")
        pool_4 = tf.layers.max_pooling2d(drop_12, [2, 2], [2, 2], padding="same")


        print("\n\nconv2d(pool_4, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(pool_4, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_13 = tf.layers.conv2d(pool_4, 512, [3, 3], [1, 1], padding="same")
        lrelu_13 = leaky_relu(x_13, alpha)
        bn_13 = tf.layers.batch_normalization(lrelu_13, training=True)
        drop_13 = tf.layers.dropout(bn_13, rate=drop_rate)


        print("\n\nconv2d(drop_13, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_13, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_14 = tf.layers.conv2d(drop_13, 512, [3, 3], [1, 1], padding="same")
        lrelu_14 = leaky_relu(x_14, alpha)
        bn_14 = tf.layers.batch_normalization(lrelu_14, training=True)
        drop_14 = tf.layers.dropout(bn_14, rate=drop_rate)


        print("\n\nconv2d(drop_14, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_14, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_15 = tf.layers.conv2d(drop_14, 512, [3, 3], [1, 1], padding="same")
        lrelu_15 = leaky_relu(x_15, alpha)
        bn_15 = tf.layers.batch_normalization(lrelu_15, training=True)
        drop_15 = tf.layers.dropout(bn_15, rate=drop_rate)


        print("\n\nconv2d(drop_15, 512, [3, 3], [1, 1], padding='same')"+
              "\nLeaky_relu"+
              "\nBatch normalization"+
              "\nDropout")
    
        result_file.write("\n\nconv2d(drop_15, 512, [3, 3], [1, 1], padding='same')"+
                          "\nLeaky_relu"+
                          "\nBatch normalization"+
                          "\nDropout")
        x_16 = tf.layers.conv2d(drop_15, 512, [3, 3], [1, 1], padding="same")
        lrelu_16 = leaky_relu(x_16, alpha)
        bn_16 = tf.layers.batch_normalization(lrelu_16, training=True)
        drop_16 = tf.layers.dropout(bn_16, rate=drop_rate)

        print("\n\nmax_pooling2d(drop_16, [ 2, 2], [2, 2], padding='same')")
        result_file.write("\n\nmax_pooling2d(drop_16, [ 2, 2], [2, 2], padding='same')")
        pool_5 = tf.layers.max_pooling2d(drop_16, [ 2, 2], [2, 2], padding="same")


        print("\n\nFully connected layer\n\nSoftMax\n")
        result_file.write("\n\nFully connected layer \n\nSoftMax\n")
        features = tf.reduce_mean(pool_5, (1, 2))
        class_logits = tf.layers.dense(features, num_classes + extra_class)
        if extra_class:
            real_class_logits, fake_class_logits = tf.split(class_logits, [num_classes, 1], 1)
            assert fake_class_logits.get_shape()[1] == 1, fake_class_logits.get_shape()
            fake_class_logits = tf.squeeze(fake_class_logits)
        else:
            real_class_logits = class_logits
            fake_class_logits = 0.

        mx = tf.reduce_max(real_class_logits, 1, keep_dims=True)
        stable_real_class_logits = real_class_logits - mx

        gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_real_class_logits), 1)) + tf.squeeze(mx) - fake_class_logits

        out = tf.nn.softmax(class_logits)
        print("\n###################################################\n")
        result_file.write("\n###################################################\n")
        return out, class_logits, gan_logits, features




    ################ Generator #####################
def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
    print("\n###################################################\n")
    print("\nGenerator Architecture")
    result_file.write("\n###################################################\n")
    result_file.write("\nGenerator Architecture")
    
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        print("\nFC_1\nbatch_normalization\nLeaky Relu")
        result_file.write("\nFC_1\nbatch_normalization\nLeaky Relu")
        x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)
        
        print("\nDeconvolution\nbatch_normalization\nLeaky Relu")
        result_file.write("\nDeconvolution\nbatch_normalization\nLeaky Relu")
        x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        
        print("\nDeconvolution\nbatch_normalization\nLeaky Relu")
        result_file.write("\nDeconvolution\nbatch_normalization\nLeaky Relu")
        x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        
        # Output layer
        print("\nDeconvolution\nActivation(Tanh)\n")
        result_file.write("\nDeconvolution\nActivation(Tanh)\n")
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')
        
        out = tf.tanh(logits)
        print("\n###################################################\n")
        result_file.write("\n###################################################\n")
        return out




##############################################
def model_loss(input_real, input_z, output_dim, y, num_classes, label_mask, alpha=0.2, drop_rate=0.5):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param output_dim: The number of channels in the output image
    :param y: Integer class labels
    :param num_classes: The number of classes
    :param alpha: The slope of the left half of leaky ReLU activation
    :param drop_rate: The probability of dropping a hidden unit
    :return: A tuple of (discriminator loss, generator loss)
    """

    # These numbers multiply the size of each layer of the generator and the discriminator,
    # respectively. You can reduce them to run your code faster for debugging purposes.
    g_size_mult = 32
    d_size_mult = 32

    # Here we run the generator and the discriminator
    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
    d_on_data = discriminator(input_real, alpha=alpha, drop_rate=drop_rate, num_classes=num_classes, size_mult=d_size_mult)
    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data
    d_on_samples = discriminator(g_model, reuse=True, alpha=alpha, drop_rate=drop_rate,num_classes=num_classes, size_mult=d_size_mult)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples

    # Here we compute `d_loss`, the loss for the discriminator.
    # This should combine two different losses:
    #  1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
    #     real-vs-fake classification problem.
    #  2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
    #     for the multi-class softmax. For this one we use the labels. Don't forget to ignore
    #     use `label_mask` to ignore the examples that we are pretending are unlabeled for the
    #     semi-supervised learning problem.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                labels=tf.ones_like(gan_logits_on_data)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                labels=tf.zeros_like(gan_logits_on_samples)))
    y = tf.squeeze(y)
    class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                                  labels=tf.one_hot(y, num_classes + extra_class,
                                                                                    dtype=tf.float32))
    class_cross_entropy = tf.squeeze(class_cross_entropy)
    label_mask = tf.squeeze(tf.to_float(label_mask))
    d_loss_class = tf.reduce_sum(label_mask * class_cross_entropy) / tf.maximum(1., tf.reduce_sum(label_mask))
    d_loss = d_loss_class + d_loss_real + d_loss_fake

    # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
    # This loss consists of minimizing the absolute difference between the expected features
    # on the data and the expected features on the generated samples.
    # This loss works better for semi-supervised learning than the tradition GAN losses.
    data_moments = tf.reduce_mean(data_features, axis=0)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
    eq = tf.equal(tf.squeeze(y), pred_class)
    correct = tf.reduce_sum(tf.to_float(eq))
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

    return d_loss, g_loss, correct, masked_correct, g_model

###########################
def model_opt(d_loss, g_loss, learning_rate, beta1, update_rate):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and biases to update. Get them separately for the discriminator and the generator
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    for t in t_vars:
        assert t in d_vars or t in g_vars

    # Minimize both players' costs simultaneously
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    for x in range(0, 3):
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

    return d_train_opt, g_train_opt, shrink_lr

##############################################
class GAN:
    """
        A GAN model.
        :param real_size: The shape of the real data.
        :param z_size: The number of entries in the z code vector.
        :param learnin_rate: The learning rate to use for Adam.
        :param num_classes: The number of classes to recognize.
        :param alpha: The slope of the left half of the leaky ReLU activation
        :param beta1: The beta1 parameter for Adam.
        """
    
    def __init__(self, real_size, z_size, learning_rate, num_classes=20, alpha=0.2, beta1=0.5, update_rate=1):
        tf.reset_default_graph()
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.input_real, self.input_z, self.y, self.label_mask = model_inputs(real_size, z_size)
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
        
        loss_results = model_loss(self.input_real, self.input_z,
                                  real_size[2], self.y, num_classes, label_mask=self.label_mask,
                                  alpha=0.2,
                                  drop_rate=self.drop_rate)
                                  self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples = loss_results
                                  
        self.d_opt, self.g_opt, self.shrink_lr = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1, update_rate)


##############################################
def train(net, dataset, epochs, batch_size, z_size, figsize=(5, 5)):
    saver = tf.train.Saver()
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    samples, train_accuracies, test_accuracies = [], [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            print("Epoch", e)
            result_file.write("\nEpoch" + str(e))

            t1e = time.time()
            num_examples = 0
            num_correct = 0
            for x, y, label_mask in dataset.batches(batch_size):
                assert 'int' in str(y.dtype)
                steps += 1
                num_examples += label_mask.sum()

                # Sample random noise for G
                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                # Run optimizers
                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y: y, net.label_mask: label_mask})
                t2 = time.time()
                num_correct += correct

            sess.run([net.shrink_lr])

            train_accuracy = num_correct / float(num_examples)

            print("\t\tClassifier train accuracy: ", train_accuracy)
            result_file.write("\n\t\tClassifier train accuracy: " + str(train_accuracy))
            num_examples = 0
            num_correct = 0
            for x, y in dataset.batches(batch_size, which_set="test"):
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]

                correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                              net.y: y,
                                                              net.drop_rate: 0.})
                num_correct += correct

            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            result_file.write("\n\t\tClassifier test accuracy: " + str(test_accuracy))
            print("\t\tStep time: ", t2 - t1)
            result_file.write("\n\t\tStep time: " + str(t2 - t1))
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)
            result_file.write("\n\t\tEpoch time: " + str(t2e - t1e))

            gen_samples = sess.run(
                net.samples,
                feed_dict={net.input_z: sample_z})
            samples.append(gen_samples)
            # _ = view_samples(-1, samples, 5, 10, figsize=figsize)
            # plt.show()

            # Save history of accuracies to view after training
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        saver.save(sess, './checkpoints/generator.ckpt')

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return train_accuracies, test_accuracies, samples



####MAIN####
def main():
    parser = argparse.ArgumentParser(description='Model parameters.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='Top_10', choices=['Top_10', 'Top_20', 'Top_31', 'Top_42'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--update_rate', type=int, default=1, help='How many times do you update the generator compared to the discriminator')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--z_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10, choices=[10, 20, 31, 42])
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--input_width', type=int, default=32)
    parser.add_argument('--input_height', type=int, default=32)
    parser.add_argument('--input_depth', type=int, default=3)
    args = parser.parse_args()
    
    
    global result_file
    temp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    result_file = open(args.out_dir + "/"+temp+".text", 'w')
    result_file.write("########### Model Settings & Parameters ###########")
    result_file.write("\nBatch_size       : "  + str(args.batch_size))
    result_file.write("\nEpochs           : "  + str(args.epochs))
    result_file.write("\nLearning Rate    : "  + str(args.learning_rate))
    result_file.write("\nDataset          : "  + str(args.dataset))
    result_file.write("\nZ size           : "  + str(args.z_size))
    result_file.write("\nReal size        : (" + str(args.input_width) + ", " + args.input_height + ", " + args.input_depth + ")")
    result_file.write("\nNumber of classes: "  + str(args.num_classes))
    result_file.write("\nDropout rate     : "  + str(args.dropout_rate))
    result_file.write("\nData Directory   : "  + str(args.data_dir))
    
    result_file.write("\n###################################################\n")
    
    #Create and load data
    load_create_data(args.data_dir)
    real_size = (args.input_width, args.input_height, args.input_depth)
    net = GAN(real_size, args.z_size, args.learning_rate, args.num_classes, args.alpha, update_rate=args.update_rate)
    
    global dataset
    dataset = Dataset(trainset, testset)
    #Delete the figure thing
    train_accuracies, test_accuracies, samples = train(net, dataset, args.epochs, args.batch_size, args.z_size, figsize=(10, 5))
    
    result_file.close()

if __name__ == "__main__":
    main()
