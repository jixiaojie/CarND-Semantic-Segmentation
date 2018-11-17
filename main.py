#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    #vgg_tag = 'vgg16'
    #vgg_input_tensor_name = 'image_input:0'
    #vgg_keep_prob_tensor_name = 'keep_prob:0'
    #vgg_layer3_out_tensor_name = 'layer3_out:0'
    #vgg_layer4_out_tensor_name = 'layer4_out:0'
    #vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #return None, None, None, None, None

    vgg_tag = 'vgg16'
    vgg = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = sess.graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor_name = sess.graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor_name = sess.graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor_name = sess.graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor_name = sess.graph.get_tensor_by_name('layer7_out:0')

    return vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name, vgg_layer7_out_tensor_name

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #return None

    reg_scale = 1e-3
    
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name= 'pool3_out_scaled')
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')
    
    output_3 = tf.layers.conv2d(pool3_out_scaled, int(4096 / 8), 1, padding = 'same', name = "output_3")
    output_4 = tf.layers.conv2d(pool4_out_scaled, int(4096 / 2), 1, padding = 'same', name = "output_4")

    output_7_01 = tf.layers.conv2d_transpose(vgg_layer7_out, int(4096 / 2), 4, (2, 2), padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = 'output_7_01')
    conv_output_7_01 = tf.layers.conv2d(output_7_01, int(4096 / 2), 1, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = "conv_output_7_01")
    add_conv_output_7_01 = tf.add(output_4, conv_output_7_01, name = "add_conv_output_7_01")

    output_7_02 = tf.layers.conv2d_transpose(add_conv_output_7_01, int(4096 / 8), 4, (2, 2), padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = 'output_7_02')
    conv_output_7_02 = tf.layers.conv2d(output_7_02, int(4096 / 8), 1, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = "conv_output_7_02")
    add_conv_output_7_02 = tf.add(output_3, conv_output_7_02, name = "add_conv_output_7_02")

    output_7_03 = tf.layers.conv2d_transpose(add_conv_output_7_02, int(4096 / 64), 4, (2, 2), padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = 'output_7_03')
    conv_output_7_03 = tf.layers.conv2d(output_7_03, int(4096 / 16), 1, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = "conv_output_7_03")

    output_7_04 = tf.layers.conv2d_transpose(conv_output_7_03, int(4096 / 128) , 4, (2, 2), padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = 'output_7_04')
    conv_output_7_04 = tf.layers.conv2d(output_7_04, int(4096 / 128), 1, padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = "add_conv_output_7_03")

    output_7_05 = tf.layers.conv2d_transpose(conv_output_7_04, num_classes, 4, (2, 2), padding = 'same', kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_scale), name = 'output_7_05')

    return output_7_05 

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    #return None, None, None

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = "logits")
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label), name = "cross_entropy_loss")
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    init = tf.global_variables_initializer()
    sess.run(init)    	
    num = 1
	
    for ep in range(epochs):
        batches_nn = get_batches_fn(batch_size)
        for i in batches_nn:
            x = i[0]
            y = i[1]

            feed_dict = {input_image: x, correct_label:y , keep_prob: 0.8}
            _, train_loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            print("Step: %d, Train_loss:%g" % (num, train_loss))
            num += 1

         
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    tests.test_for_kitti_dataset(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    batch_size = 10
    learning_rate = 1e-6
    epochs = 32

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 2], name="correct_label")
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, './data/vgg')
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        print("Geting training accuracy ...")
        batches_nn = get_batches_fn(batch_size)
        num = 1
        for i in batches_nn:
            x = i[0]
            y = i[1]

            feed_dict = {input_image: x, correct_label:y , keep_prob: 1.0}
            y = y.astype(np.float32)
            y.reshape(-1,2)
            y_pre = sess.run(logits, feed_dict=feed_dict)
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(tf.reshape(y,(-1, 2)),1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(num, "  accuracy:", sess.run(accuracy, feed_dict=feed_dict))
            num += 1

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
