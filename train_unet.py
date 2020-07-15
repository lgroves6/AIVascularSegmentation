import numpy as np
from unet import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
import argparse
import datetime


# parser = argparse.ArgumentParser()
# parser.add_argument('--datapath', type=str, default='/home/bvanberl/scratch/data/')
# args = parser.parse_args()

# Load US images and segmentation labels
# data_path = args.datapath
data_path = '/scratch/bvanberl/data/'
X = np.load(data_path + 'X.npy')
Y = np.load(data_path + 'Y.npy')
print("SHAPES:",X.shape,Y.shape)
X = np.load(data_path + 'X_crossval.npy')
Y = np.load(data_path + 'Y_crossval.npy')
print("SHAPES:",X.shape,Y.shape)
X = np.squeeze(X, axis=1)
Y = np.squeeze(Y, axis=1)
X = np.expand_dims(X, axis=3)
Y = np.expand_dims(Y, axis=3)
Y = to_categorical(Y)
print("Y shape = " + str(Y.shape))
#X_train, X_valid, Y_train, Y_valid = train_test_split(X[:-434], Y[:-434], test_size=0.1, random_state=2019)

'''
# Training set will be users 2-4, test set will be user 1 (idx=0-203), val set will be user 5 (idx=-434)
X_val = X[-434:]
Y_val = Y[-434:]
X_test = X[0:203]
Y_test = Y[0:203]
X_train = X[203:-434]
Y_train = Y[203:-434]
print("Train set 203:-434 (user 2-4), val set -434: (user 5), test set 0:203 (user 1)")
'''

########################################################
# PARTITION FOR CROSS-VALIDATION
'''
print("SET A")
permut = list(range(0,78)) + list(range(611,1865)) + list(range(1955,2215)) + list(range(2314,2439)) # train
permut += list(range(1865,1955)) + list(range(2215,2295)) # val
permut += list(range(78,611)) + list(range(2295,2314)) # test
train_start = 0
train_end = 1717
val_start = 1717
val_end = 1887
test_start = 1887
test_end = 2439

print("SET B")
permut = list(range(0,1012)) + list(range(1432,1714)) + list(range(1792,2021)) + list(range(2108,2314)) + list(range(2369,2439)) # train
permut += list(range(1714,1792)) + list(range(2021,2108)) # val
permut += list(range(1012,1432)) + list(range(2314,2369)) # test
train_start = 0
train_end = 1799
val_start = 1799
val_end = 1964
test_start = 1964
test_end = 2439

print("SET C")
permut = list(range(78,1432)) + list(range(1714,2108)) + list(range(2215,2369)) # train
permut += list(range(0,78)) + list(range(2108,2215)) # val
permut += list(range(1432,1714)) + list(range(2369,2439)) # test
train_start = 0
train_end = 1902
val_start = 1902
val_end = 2087
test_start = 2087
test_end = 2439
'''
print("SET D")
permut = list(range(0,611)) + list(range(1012,1792)) + list(range(1865,1955)) + list(range(2021,2215)) + list(range(2295,2439)) # train
permut += list(range(1792,1865)) + list(range(1955,2021)) # val
permut += list(range(611,1012)) + list(range(2215,2295)) # test
train_start = 0
train_end = 1819
val_start = 1819
val_end = 1958
test_start = 1958
test_end = 2439

assert len(permut) == X.shape[0]
permut = np.array(permut)
X = X[permut]
Y = Y[permut]
X /= 255.0

X_test = X[test_start:test_end]
Y_test = Y[test_start:test_end]
X_val = X[val_start:val_end]
Y_val = Y[val_start:val_end]
X_train = X[train_start:train_end]
Y_train = Y[train_start:train_end]

########################################################

# Define callbacks
callbacks = [
    EarlyStopping(patience=15, verbose=1),
    TensorBoard(log_dir='./logs'),
    ReduceLROnPlateau(factor=0.75, patience=15, min_lr=0.000005, verbose=1),
    ModelCheckpoint('./models/setD_model_ckpt.h5', verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss')
]

# Initialize model
NUM_GPUS = 2
model, parallel_model = unet(num_gpus=NUM_GPUS)

# Train the model. 90% train and 10% validation
#model.fit(X_train, Y_train, batch_size=16, epochs=75, verbose=1,validation_data=(X_valid, Y_valid), shuffle=True, callbacks=callbacks)


data_gen_args = dict(
    rotation_range=15,
    shear_range=0.05,
    zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 209
batch_size = 32
#image_datagen.fit(X_train, augment=True, seed=seed)
#mask_datagen.fit(Y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=batch_size)
mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=batch_size)

def TrainGenerator(train_generator,train_generator1):
    while True:
        xy = train_generator.next()
        xy1 = train_generator1.next()
        yield (xy,xy1)
train_generator = TrainGenerator(image_generator, mask_generator)

#class_weight = class_weight.compute_class_weight('balanced', np.unique(np.ravel(Y_train,order='C')), np.ravel(Y_train,order='C'))

#print(model.summary())
parallel_model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, callbacks=callbacks, validation_data=(X_val, Y_val))

# Evaluate performance on test set
preds = parallel_model.evaluate(x=X_test, y=Y_test) # Evaluate model's performance on the test set
print("Test set loss = " + str(preds[0]))
print("Test set accuracy = " + str(preds[1]))
print("Test set dice = " + str(preds[2]))
print("Test set recall = " + str(preds[3]))
print("Test set precision = " + str(preds[4]))
print("Test set specificity = " + str(preds[5]))

# Save model weights
if NUM_GPUS >= 2:
    model.save('./models/setD_model_unet.h5')
else:
    parallel_model.save('./models/model_unet' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')

