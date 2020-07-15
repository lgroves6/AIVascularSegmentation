import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from unet import *
from tensorflow.keras.backend import set_session
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import *


def softmax_output_to_one_hot(y):
    max_indices = np.argmax(y, axis=2)
    return np.eye(np.max(max_indices) + 1)[max_indices]


'''
Displays the ultrasound image and its true and predicted segmentations. Colours are as follows:
Green = US image
Red = Ground truth segmentation
Blue = Predicted segmentation
PARAMS:
- idx: the index in the dataset of the image to display
- y: the predicted segmentation of the image at idx
'''
def display_image(idx, y):
    x = X[idx]
    y_pred_0 = y[:,:,1]
    y_pred_1 = y[:,:,2]
    #y = np.argmax(y, axis=2)
    print(y[0][0])
    #img = np.squeeze(np.maximum(x, y), axis=2)
    y_true_col = 0.5*np.stack((np.squeeze(Y[idx], axis=2),np.zeros((y.shape[0],y.shape[1])), np.zeros((y.shape[0],y.shape[1]))), axis=2) # RED
    y_pred_col_0 = 0.5*np.stack((np.zeros((y.shape[0], y.shape[1])), np.zeros((y.shape[0], y.shape[1])),  y_pred_1), axis=2) # BLUE
    y_pred_col_1 = 0.5*np.stack((y_pred_0, y_pred_0, np.zeros((y.shape[0], y.shape[1]))), axis=2)  # YELLOW
    xcol = np.stack((np.squeeze(x, axis=2), np.squeeze(x, axis=2), np.squeeze(x, axis=2)), axis=2) # WHITE
    plt.clf()
    plt.imshow(xcol + y_pred_col_0 + y_pred_col_1 + y_true_col)  # Display the image
    plt.title(str(idx) + ' / ' + str(Y.shape[0] - 1))
    plt.text(-100, 50, "US image", color="white")
    plt.text(-100, 90, "ground truth", color="red")
    plt.text(-100, 130, "JV prediction", color="blue")
    plt.text(-100, 150, "CA prediction", color="yellow")
    fig.canvas.draw()
    plt.show()


def make_prediction(idx):
    if model_name == 'unet':
        y = model.predict(np.expand_dims(X[idx], axis=0))
    else:
        if (idx < 2): idx = 2
        x = np.concatenate([np.expand_dims(X[idx], axis=0),
                                          np.expand_dims(X[idx-1], axis=0),
                                          np.expand_dims(X[idx-2], axis=0)], axis=0)
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)
    y = np.squeeze(y, axis=0)
    y = softmax_output_to_one_hot(y)
    display_image(idx, y)
    print("Dice = ", get_gen_dice_score(Y[idx], y))


def evaluate_set():
    X_test = X[0:110]
    Y_test = Y[0:110]
    #model.compile(optimizer=Adam(lr=1e-4), loss=generalized_dice_loss, metrics=['accuracy'])
    #preds = model.evaluate(x=X_test, y=Y_test)
    avg_dice = 0
    for i in range(0, X_test.shape[0]):
        y = model.predict(np.expand_dims(X[i], axis=0))
        y = np.squeeze(y, axis=0)
        y = softmax_output_to_one_hot(y)
        avg_dice += get_gen_dice_score(Y_test[i], y)
    avg_dice /= X_test.shape[0]
    print("Avg test set Dice Score = " + str(avg_dice))


def key_pressed(event):
    if event.key == 'c':
        globals()['idx'] = (idx + 1)%X.shape[0]
        make_prediction(idx)
    elif event.key == 'z':
        globals()['idx'] = (idx - 1)%X.shape[0]
        make_prediction(idx)

'''
Calculates the dice similarity coefficient of two segmentations
PARAMS:
- y_true: the ground truth segmentation
- y_pred: the prediced segmentation
'''
def get_gen_dice_score(y_true, y_pred):
    epsilon = 1e-5 # To ensure no division by 0
    #y_pred = tf.math.round(y_pred) # y_pred was a sigmoid activation
    numerator = denominator = epsilon
    y_true = to_categorical(y_true)
    intersection = y_true * y_pred
    union = y_true + y_pred
    for i in range(0, y_pred.shape[-1]):
        intersection_sum = np.sum(intersection[...,i])
        union_sum = np.sum(union[...,i])
        class_weight = 1.0 / (np.sum(y_true[...,i]) ** 2 + epsilon)
        numerator += class_weight * intersection_sum
        denominator += class_weight * union_sum
    dice_score = 2 * numerator / denominator
    return dice_score



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model_name = 'unet30'
X = np.load('./datasets/X.npy')
Y = np.load('./datasets/Y.npy')
model = load_model('./models/model_' + model_name + '.h5',
                   custom_objects={'generalized_dice_loss': generalized_dice_loss, 'generalized_dice_score': generalized_dice_score,
                                   'custom_loss': custom_loss})
idx = 0
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', key_pressed)
make_prediction(idx)