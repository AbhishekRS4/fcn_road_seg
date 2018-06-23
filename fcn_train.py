# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from fcn_utils import init, read_config_file, get_data, preprocess_images, get_train_test_set
import network_architecture as na

param_config_file_name = os.path.join(os.getcwd(), "fcn_config.json")


# return the softmax layer
def get_softmax_layer(input_tensor, dim = -1, name = 'softmax'):
    prediction = tf.nn.softmax(input_tensor, dim = dim, name = name)
    return prediction


# return the sorensen-dice coefficient
def dice_loss(ground_truth, predicted_logits, dim = -1, smooth = 1e-5, name = 'mean_dice_loss'):
    predicted_probs = get_softmax_layer(input_tensor = predicted_logits, dim = dim)
    intersection = tf.reduce_sum(tf.multiply(ground_truth, predicted_probs), axis = [1, 2, 3])
    union = tf.reduce_sum(ground_truth, axis = [1, 2, 3]) + tf.reduce_sum(predicted_probs, axis = [1, 2, 3])
    dice_coeff = tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
     
    dice_loss = -tf.log(dice_coeff, name = name)
    return dice_loss


# return cross entropy loss
def cross_entropy_loss(ground_truth, prediction, axis, name = "mean_cross_entropy"):
    mean_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth, logits = prediction, dim = axis), name = name)
    return mean_ce


# return the optimizer which has to be used to minimize the loss function
def get_optimizer(learning_rate, loss_function):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_function)
    return adam_optimizer


# return the placeholder
def get_placeholders(img_placeholder_shape, training = False, mask_placeholder_shape = None):
    img_pl = tf.placeholder(tf.float32, shape = img_placeholder_shape)
    # set the image placeholder

    if training:
        # set the label placeholder in the training phase
        mask_pl = tf.placeholder(tf.float32, shape = mask_placeholder_shape)
        return (img_pl, mask_pl)

    return img_pl


# save the trained model
def save_model(session, model_directory, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(model_directory, model_file)), global_step = (epoch + 1))


def batch_train():

    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    model_to_use = config['model_to_use']
    print("Reading the config file completed........")
    print("")

    print("Initializing.............................")
    model_directory = config['model_directory'][model_to_use] + str(config['num_epochs'])
    init(model_directory)
    print("Initializing completed...................")
    print("")

    print("Reading train data.......................")
    all_images_list = os.listdir(config['INPUTS_PATH'])

    train_valid_list, test_list = get_train_test_set(all_images_list, test_size = 0.5)     
    train_list, valid_list = get_train_test_set(train_valid_list, test_size = 0.04) 

    train_images, train_masks = get_data(train_list, config['INPUTS_PATH'], config['MASKS_PATH'], config['TARGET_IMAGE_SIZE'] + [config['num_classes']])
    valid_images, valid_masks = get_data(valid_list, config['INPUTS_PATH'], config['MASKS_PATH'], config['TARGET_IMAGE_SIZE'] + [config['num_classes']])

    print("Reading train data completed.............")
    print("")

    print("Preprocessing the data...................")
    train_images = preprocess_images(train_images)
    valid_images = preprocess_images(valid_images)
    print("Preprocessing of the data completed......")
    print("")


    print("Building the network.....................")
    axis = -1
    if config['data_format'] == 'channels_last': 
        IMAGE_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['NUM_CHANNELS']]
        MASK_PLACEHOLDER_SHAPE = [None] + config['TARGET_IMAGE_SIZE'] + [config['num_classes']]
    else:
        IMAGE_PLACEHOLDER_SHAPE = [None] + [config['NUM_CHANNELS']] + config['TARGET_IMAGE_SIZE']
        MASK_PLACEHOLDER_SHAPE = [None] + [config['num_classes']] + config['TARGET_IMAGE_SIZE']
        train_images = np.transpose(train_images, [0, 3, 1, 2])
        train_masks = np.transpose(train_masks, [0, 3, 1, 2])
        valid_images = np.transpose(valid_images, [0, 3, 1, 2])
        valid_masks = np.transpose(valid_masks, [0, 3, 1, 2])
        axis = 1 
    
    img_pl, mask_pl = get_placeholders(img_placeholder_shape = IMAGE_PLACEHOLDER_SHAPE, training = bool(config['TRAINING']), mask_placeholder_shape = MASK_PLACEHOLDER_SHAPE)
    net_arch = na.FCN(config['VGG_PATH'], config['data_format'], config['num_classes'])
    net_arch.vgg_encoder(img_pl)
 
    if model_to_use % 3 == 0: 
        net_arch.fcn_8()
        logits = net_arch.logits
    elif model_to_use % 3 == 1:
        net_arch.fcn_16()
        logits = net_arch.logits
    else:
        net_arch.fcn_32()
        logits = net_arch.logits

    if model_to_use == 0 or model_to_use == 1 or model_to_use == 2:
        loss = cross_entropy_loss(mask_pl, logits, axis = axis)
    elif model_to_use == 3 or model_to_use == 4 or model_to_use == 5:
        loss = dice_loss(mask_pl, logits, axis = axis)
    else:
        loss_1 = dice_loss(mask_pl, logits, axis = axis)
        loss_2 = cross_entropy_loss(mask_pl, logits, axis = axis)
        loss = loss_1 + loss_2
 
    optimizer = get_optimizer(config['learning_rate'], loss)
 
    print("Building the network completed...........")
    print("")
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = int(math.ceil(train_images.shape[0] / float(batch_size)))
    
    print("Train data shape : " + str(train_images.shape))
    print("Train mask shape : " + str(train_masks.shape))
    print("")
    print("Validation data shape : " + str(valid_images.shape))
    print("Validation mask shape : " + str(valid_masks.shape))
    print("")
    print("Number of epochs to train : " + str(num_epochs))
    print("Batch size : " + str(batch_size))
    print("Number of batches : " + str(num_batches))
    print("")

    print("Training the Network.....................")
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()
    
    for epoch in range(num_epochs):
        ti = time.time()
        temp_loss_per_epoch = 0
        train_images, train_masks = shuffle(train_images, train_masks) 
        for batch_id in range(num_batches):
            batch_images = train_images[batch_id * batch_size : (batch_id + 1) * batch_size]
            batch_masks = train_masks[batch_id * batch_size : (batch_id + 1) * batch_size]
            
            _, loss_per_batch = ss.run([optimizer, loss], feed_dict = {img_pl : batch_images, mask_pl : batch_masks})
            temp_loss_per_epoch += (batch_images.shape[0] * loss_per_batch)
        ti = time.time() - ti
        loss_validation_set = ss.run(loss, feed_dict = {img_pl : valid_images, mask_pl : valid_masks})
        train_loss_per_epoch.append(temp_loss_per_epoch)
        valid_loss_per_epoch.append(loss_validation_set)

        print("Epoch : " + str(epoch+1) + "/" + str(num_epochs) + ", time taken : " + str(ti) + " sec.")
        print("Avg. training loss : " + str(temp_loss_per_epoch / train_images.shape[0]))
        print("Avg. validation loss : " + str(loss_validation_set))
        print("")

        if (epoch + 1) % 25 == 0:
            save_model(ss, model_directory, config['model_file'][model_to_use], epoch)
        

    print("Training the Network Completed...........")
    print("")
    
    print("Saving the model.........................")
    save_model(ss, model_directory, config['model_file'][model_to_use % 3], epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)
    
    train_loss_per_epoch = np.true_divide(train_loss_per_epoch, train_images.shape[0])
   
    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, config['model_metrics'][model_to_use % 3])), (losses_dict))
    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, "train_list.npy")), np.array(train_list))
    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, "valid_list.npy")), np.array(valid_list))

    print("Saving the model Completed...............")
    print("")

    ss.close()

def main():
    batch_train()

if __name__ == '__main__':
    main()

