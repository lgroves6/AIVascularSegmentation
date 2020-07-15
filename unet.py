from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model, to_categorical


def custom_loss_layer(layer):
    def loss(y_true, y_pred):
        w = tf.math.sqrt(1.0 - layer)
        return K.mean(K.binary_crossentropy(y_true, y_pred) * w, axis=-1)
    return loss


def generalized_dice_score(y_true, y_pred):
    epsilon = 1e-5  # To ensure no division by 0
    numerator = denominator = epsilon
    intersection = y_true * y_pred
    union = y_true + y_pred
    for i in range(0, y_pred.shape[-1]):
        intersection_sum = tf.reduce_sum(intersection[..., i])
        union_sum = tf.reduce_sum(union[..., i])
        class_weight = 1.0 / (tf.reduce_sum(y_true[..., i]) ** 2 + epsilon)
        numerator += class_weight * intersection_sum
        denominator += class_weight * union_sum
    dice_score = 2 * numerator / denominator
    return dice_score
	
	
def recall(y_true, y_pred):
    # Compute average recall over segmentation classes
    avg_recall = 0 # NOTE: sensitivity = recall
    num_classes = y_pred.shape[-1]
    for i in range(0, num_classes):
        true_pos = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        true_neg = tf.reduce_sum((1 - y_true[..., i]) * (1 - y_pred[..., i]))
        false_neg = tf.reduce_sum(1 - y_pred[..., i]) - true_neg
        recall = true_pos / (true_pos + false_neg)     # i.e. recall
        avg_recall += recall
    avg_recall /= num_classes.value
    return avg_recall
	
def specificity(y_true, y_pred):
    # Compute average specificity over segmentation classes
    avg_specificity = 0
    num_classes = y_pred.shape[-1]
    for i in range(0, num_classes):
        true_pos = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        true_neg = tf.reduce_sum((1 - y_true[..., i]) * (1 - y_pred[..., i]))
        false_pos = tf.reduce_sum(y_pred[..., i]) - true_pos
        specificity = true_neg / (true_neg + false_pos)
        avg_specificity += specificity
    avg_specificity /= num_classes.value
    return avg_specificity

def precision(y_true, y_pred):
    # Compute average precision over segmentation classes
    num_classes = y_pred.shape[-1]
    avg_precision = 0
    for i in range(0, num_classes):
        true_pos = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        false_pos = tf.reduce_sum(y_pred[..., i]) - true_pos
        precision = true_pos / (true_pos + false_pos)
        avg_precision += precision
    avg_precision /= num_classes.value
    return avg_precision


def generalized_dice_loss(y_true, y_pred):
    dice_score = generalized_dice_score(y_true, y_pred)
    return 1.0 - dice_score


def custom_loss(y_true, y_pred):
    return generalized_dice_loss(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)


'''
def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    x = 6

    conv1 = Conv2D(2**x, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(2**x, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2**(x+1), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2**(x+1), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(2**(x+2), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(2**(x+2), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(2**(x+3), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(2**(x+3), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(2**(x+4), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(2**(x+4), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(2**(x+3), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(2**(x+3), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(2**(x+3), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(2**(x+2), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(2**(x+2), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(2**(x+2), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(2**(x+1), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(2**(x+1), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2**(x+1), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(2**x, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(2**x, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(2**x, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
'''

def unet(pretrained_weights=None, input_size=(256, 256, 1), num_gpus=1):
    num_classes = 3
    x = 6
    inputs = Input(input_size)
    main_path = inputs
    num_convs = 3
    res_layers = [None] * num_convs

    # Down convolutions
    for i in range(0, num_convs):
        main_path = Conv2D(2 ** (x + i), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
        main_path = Conv2D(2 ** (x + i), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
        #if i == 3:
        #    main_path = Dropout(0.5)(main_path)
        res_layers[i] = main_path
        main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Bottleneck
    main_path = Conv2D(2 ** (x + num_convs), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
    main_path = Conv2D(2 ** (x + num_convs), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
    #main_path = Dropout(0.5)(main_path)

    # Up convolutions
    for i in reversed(range(0, num_convs)):
        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = Conv2D(2 ** (x + i), 2, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
        main_path = concatenate([res_layers[i], main_path], axis=3)
        main_path = Conv2D(2 ** (x + i), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)
        main_path = Conv2D(2 ** (x + i), 3, activation='relu', padding='same', kernel_initializer='he_normal')(main_path)

    # Output
    output = Conv2D(num_classes, 1, activation='softmax')(main_path)

    # Define model
    model = Model(inputs=inputs, outputs=output)
    parallel_model = model

    # Replicate the model on 2 GPUs
    if num_gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=num_gpus)

    # Set optimizer, loss function
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy', generalized_dice_score, recall, precision, specificity])
    #print(model.summary())

    # Load pretrained weights, if provided
    if (pretrained_weights):
        parallel_model.load_weights(pretrained_weights)

    return model, parallel_model



class Squeeze(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Squeeze, self).__init__()
    def call(self, inputs, axis=[1]):
        return tf.squeeze(inputs, axis=axis)

class Split(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Split, self).__init__()
    def call(self, inputs, num_or_size_splits=3, axis=1):
        return tf.split(inputs, num_or_size_splits=num_or_size_splits, axis=axis)

class Stack(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Stack, self).__init__()
    def call(self, inputs, axis=1):
        return tf.stack(inputs, axis=axis)



def runet(pretrained_weights=None, input_size=(3, 256, 256, 1)):
    inputs = Input(input_size)

    img0, img1, img2 = Split()(inputs, num_or_size_splits=3, axis=1)
    down_layers = [img0, img1, img2]
    for i in range(len(down_layers)):
        down_layers[i] = Squeeze()(down_layers[i], axis=[1])
    res_layers = [None] * 4
    save_layers = [None] * 3

    for i in range(len(res_layers)):
        for j in range(len(down_layers)):
            down_layers[j] = Conv2D(2 ** (i+5), 3, activation='relu', padding='same', kernel_initializer='he_normal')(down_layers[j])
            save_layers[j] = Conv2D(2 ** (i+5), 3, activation='relu', padding='same', kernel_initializer='he_normal')(down_layers[j])
            down_layers[j] = MaxPooling2D(pool_size=(2, 2))(save_layers[j])

        #conv_concat = Stack()([save_layers[0], save_layers[1], save_layers[2]], axis=1)
        #res_layers[i] = ConvLSTM2D(2 ** (i+5), 3, padding="same", return_sequences=False)(conv_concat)
        res_layers[i] = save_layers[2]

    conv_concat = Stack()([down_layers[0], down_layers[1], down_layers[2]], axis=1)
    convlstm = ConvLSTM2D(512, 3, padding="same", return_sequences=False)(conv_concat)

    bottleneck = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convlstm)
    bottleneck = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bottleneck)
    bottleneck = Dropout(0.5)(bottleneck)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(bottleneck))
    merge6 = concatenate([res_layers[3], up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(25, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([res_layers[2], up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([res_layers[1], up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([res_layers[0], up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    #print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
