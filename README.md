# AIVascularSegmentation

train_mrcc - file to train a Mask R-CNN algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation. 

train_unet - file to train a U-Net algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation. 

test_mrcnn and test_unet - files to run to obtain test results.

Unet - file that executes the U-net segmentation algorithm. 

mrcnnconfig - Mask R-CNN Base Configurations class.

mrcnnmodel - The main Mask R-CNN model implementation.

mrcnnmodeldist - delete?

mrcnnparrallel_model - Mask R-CNN Multi-GPU Support for Keras.

mrcnnsubclass - ?

mrcnnutils - Mask R-CNN Common utility functions and classes.

mrcnnvisualize - Mask R-CNN Display and Visualization Functions.

Please go to https://1drv.ms/u/s!Akxm4gUER2IFag1CJ1LtM_qB6HY?e=0aqATl to download the numpy image arrays. 
