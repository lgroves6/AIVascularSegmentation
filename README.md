# AIVascularSegmentation

Descriptions of scripts in this repository are listed below.
- [_train_mrcnn.py_](train_mrcnn.py) - file to train a Mask R-CNN algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation. 
- [_train_unet.py_](train_unet.py) - file to train a U-Net algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation. 
- [_test_mrcnn.py_](test_mrcnn.py) - Test Mask R-CNN performance, display predictions of the Mask R-CNN model
- [_test_unet.py_](test_unet.py) - Test U-Net performance, display predictions of the U-Net model
- [_unet.py_](unet.py) - U-Net model implementation
- [_mrcnnconfig.py_](mrcnnconfig.py) - Mask R-CNN base Config class
- [_mrcnnmodel.py_](mrcnnmodel.py) - Mask R-CNN model implementation
- [_mrcnnparrallel_model.py_](mrcnnparrallel_model.py) - Mask R-CNN Multi-GPU support
- [_mrcnnsubclass.py_](mrcnnsubclass.py) - Subclasses of the Mask R-CNN Config and Dataset classes
- [_mrcnnutils.py_](mrcnnutils.py) - Mask R-CNN common utility functions and classes.
- [_mrcnnvisualize.py_](rcnnvisualize.py) - Mask R-CNN display and visualization functions.

Please go to https://1drv.ms/u/s!Akxm4gUER2IFag1CJ1LtM_qB6HY?e=0aqATl to download the numpy image arrays. 

We express our gratitude to Matterport, as we extended their [Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN) to make this project possible.
