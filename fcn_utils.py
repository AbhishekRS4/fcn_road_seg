# @author : Abhishek R S

import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# the order of channels should be BGR for subtracting mean
VGG_MEAN = np.array([103.939, 116.779, 123.68]).reshape(3, 1).T

# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config


# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# return either train or test data
def get_data(images_list, images_directory, masks_directory, mask_shape):

    # all_images is a list holding all the images
    all_images = list()

    # all_labels is a list holding all the correspoding labels
    all_masks = list()

    # read the image and get the corresponding label and append it to the appropriate list
    for img_file in images_list:
        temp_img = cv2.imread(os.path.join(images_directory, img_file))
        temp_mask = cv2.imread(os.path.join(masks_directory, img_file.split('.')[0] + '_L.png'))

        temp_req_mask = np.empty(mask_shape, dtype = np.float32)
        temp_bool_mask = np.zeros(mask_shape, dtype = np.bool)

        temp_bool_mask[:, :, 1] = np.all((temp_mask[:, :] == [128, 64, 128]), axis = 2) #Road Semantic Class 
        temp_bool_mask[:, :, 1] = np.logical_or(temp_bool_mask[:, :, 1], np.all((temp_mask[:, :] == [192, 0, 128]), axis = 2)) #Lane Markings Drivable 
        temp_bool_mask[:, :, 0] = np.invert(temp_bool_mask[:, :, 1]) #Non Road Class

        temp_req_mask = 1 * temp_bool_mask

        all_images.append(temp_img)
        all_masks.append(temp_req_mask)
            
    # convert the data into numpy array and return it
    return (np.array(all_images), np.array(all_masks))


# split into train and test set
def get_train_test_split(images_list, test_size = 0.5):
    train_images_list, test_images_list = train_test_split(images_list, test_size = test_size, random_state = 4)
    
    return (train_images_list, test_images_list) 


# return the accuracy score of the predictions by the model
def get_accuracy_score(labels_groundtruth, labels_predicted):
    return accuracy_score(labels_groundtruth, labels_predicted)


def preprocess_images(all_images):
    all_images = all_images - VGG_MEAN

    return all_images 
