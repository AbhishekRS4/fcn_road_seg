# @author : Abhishek R S

import os
import time
import numpy as np
import cv2
import tensorflow as tf

import network_architecture as na
from fcn_utils import read_config_file, init, preprocess_images

param_config_file_name = os.path.join(os.getcwd(), 'fcn_config.json')
alpha = 0.2

def get_softmax_layer(logits, axis, name = 'softmax'):
    probs = tf.nn.softmax(logits, dim = axis, name = name)
    return probs

# return the placeholder
def get_placeholder(img_placeholder_shape):
    # set the image placeholder
    img_pl = tf.placeholder(tf.float32, shape = img_placeholder_shape)

    return img_pl


# run inference on test set
def infer():
    print('Reading the Config File..................')
    config = read_config_file(param_config_file_name)
    model_to_use = config['model_to_use']
    model_directory = config['model_directory'][model_to_use] + str(config['num_epochs'])
    masks_directory = 'masks'
    overlays_directory = 'overlays'
    init(os.path.join(model_directory, masks_directory))
    init(os.path.join(model_directory, overlays_directory))
    print('Reading the Config File Completed........')
    print('')

 
    print('Loading the Network.....................')
 
    axis = -1 
    if config['data_format'] == 'channels_last':
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['target_image_size'] + [config['num_channels']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['num_channels']] + config['target_image_size']
        axis = 1
 
    img_pl = get_placeholder(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE)

    net_arch = na.FCN(config['vgg_path'], config['data_format'], config['num_classes'])
    net_arch.vgg_encoder(img_pl)

    if model_to_use % 3 == 0:
        net_arch.fcn_8()
    elif model_to_use % 3 == 1:
        net_arch.fcn_16()
    else:
        net_arch.fcn_32()

    network_logits = net_arch.logits
    probs_prediction = get_softmax_layer(logits = network_logits, axis = axis)
    labels_prediction = tf.argmax(probs_prediction, axis = axis)
    print('Loading the Network Completed...........')
    print('')

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'][model_to_use % 3])) + '-' + str(config['num_epochs']))

    print('Inference Started.......................')
    for img_file in os.listdir(config['inputs_path']):
        _img = cv2.imread(os.path.join(config['inputs_path'], img_file))
        img = _img
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_images(img)
        img = np.transpose(img, [0, 3, 1, 2]) 
        ti = time.time()
        labels_predicted = ss.run(labels_prediction, feed_dict = {img_pl : img})
        ti = time.time() - ti
        print('Time Taken for Inference : ' +str(ti))
        print('')

        labels_predicted = np.transpose(labels_predicted, [1, 2, 0])
        mask = np.dot(labels_predicted, np.array(config['road_color_map']).reshape(1, 3)).astype(np.uint8)
        overlay = cv2.addWeighted(_img, 1, mask, alpha, 0, _img)
        cv2.imwrite(os.path.join(os.getcwd(), os.path.join(model_directory, os.path.join(masks_directory, 'mask_' + img_file))), mask)
        cv2.imwrite(os.path.join(os.getcwd(), os.path.join(model_directory, os.path.join(overlays_directory, 'overlay_' + img_file))), overlay)

    print('Inference Completed') 

    print('')
    ss.close()


def main():
    infer()

if __name__ == '__main__':
    main()
