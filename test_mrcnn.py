from mrcnnsubclass import VesselConfig, VesselDataset
from mrcnnmodel import *
from mrcnnvisualize import *
from mrcnnutils import *
from keras.utils import to_categorical
import sys

class InferenceConfig(VesselConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([70.0])
    BACKBONE = "resnet50"


def key_pressed(event):
    if event.key == 'c':
        globals()['test_idx'] = (test_idx + 1) % len(test_dataset.image_ids)
        test_image(test_idx)
    elif event.key == 'z':
        globals()['test_idx'] = (test_idx - 1) % len(test_dataset.image_ids)
        test_image(test_idx)


def take_best_proposals_only(r):
    idxs_to_keep = []
    classes_present = np.unique(r['class_ids'])
    print("Num predicted classes: ", len(classes_present))
    for i in range(len(classes_present)):
        max_score = 0
        idx_to_keep = 0
        for j in range(len(r['scores'])):
            if (r['scores'][j] > max_score) and (r['class_ids'][j] == classes_present[i]):
                max_score = r['scores'][j]
                idx_to_keep = j
        idxs_to_keep.append(idx_to_keep)
    r['rois'] = r['rois'][idxs_to_keep]
    r['masks'] = r['masks'][..., idxs_to_keep]
    r['class_ids'] = r['class_ids'][idxs_to_keep]
    r['scores'] = r['scores'][idxs_to_keep]
    print("SHAPE",r['class_ids'].shape,r['masks'].shape)
    if r['masks'].shape[2] == 1:
		
        if 1 not in classes_present:
            r['masks'] = np.concatenate((np.full((256,256,1), False, dtype=bool), r['masks']), axis=2)
        elif 2 not in classes_present:
            r['masks'] = np.concatenate((r['masks'], np.full((256,256,1), False, dtype=bool)), axis=2)
    elif r['masks'].shape[2] == 0:
        r['masks'] = np.full((256,256,2), False, dtype=bool)
    return r


def generalized_dice_score(y_true, y_pred):
    epsilon = 1e-5  # To ensure no division by 0
    numerator = denominator = epsilon
    y_true = np.delete(y_true, 0, 2) # Delete background
    y_pred = y_pred * 1 # Convert from bool to int
    intersection = y_true * y_pred
    union = y_true + y_pred
    for i in range(1, y_pred.shape[-1]):
        intersection_sum = np.sum(intersection[..., i])
        union_sum = np.sum(union[..., i])
        class_weight = 1.0 / (np.sum(y_true[..., i]) ** 2 + epsilon)
        numerator += intersection_sum
        denominator += union_sum
    dice_score = 2 * numerator / denominator
    return dice_score


def classification_metrics(y_true, y_pred):
    # Compute average sensitivity over segmentation classes
    print(y_true.shape, y_pred.shape)
    y_true = np.delete(y_true, 0, 2) # Delete background
    num_classes_missing = y_true.shape[2] - y_pred.shape[2]
    avg_recall = 0 # NOTE: sensitivity = recall
    avg_specificity = 0
    avg_precision = 0
    num_classes = y_true.shape[-1]
    for i in range(0, num_classes):
        true_pos = np.sum(y_true[..., i] * y_pred[..., i])
        true_neg = np.sum((1 - y_true[..., i]) * (1 - y_pred[..., i]))
        false_pos = np.sum(y_pred[..., i]) - true_pos
        false_neg = np.sum(1 - y_pred[..., i]) - true_neg
        recall = true_pos / (true_pos + false_neg)     # i.e. recall
        specificity = true_neg / (true_neg + false_pos)
        precision = true_pos / (true_pos + false_pos)
        if (recall > 1 or specificity > 1):
            print(true_pos, true_neg, false_pos, false_neg, np.sum(y_pred[..., i]), np.sum(1 - y_pred[..., i]),
                  np.sum(y_true[..., i]), np.sum(1 - y_true[..., i]))
        avg_recall += recall
        avg_specificity += specificity
        avg_precision += precision
    avg_recall /= num_classes
    avg_specificity /= num_classes
    avg_precision /= num_classes
    return recall, specificity, precision

def test_image(idx):
    image_id = test_dataset.image_ids[0] + idx
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        load_image_gt(test_dataset, inference_config,
                      image_id, use_mini_mask=False)

    # Make image prediction
    results = model.detect([original_image], verbose=1)
    r = results[0]

    # Take the region proposal with the highest confidence from each class
    r = take_best_proposals_only(r)

    # Calculate Dice score
    y_true = to_categorical(Y[test_idx])
    try:
        dice_score = generalized_dice_score(y_true, r['masks'])
    except:
        dice_score = 0
    print("DICE=", dice_score, " class_ids:",r['class_ids'])

    # Clear the graphs
    ax1.clear()
    ax2.clear()
    '''
    colours = [1, 1, 1] * len(np.unique(r['class_ids'])) # Red for carotid, blue for jugular
    ca_indices = np.squeeze(np.argwhere(r['class_ids'] == 1), axis=0)
    jv_indices = np.squeeze(np.argwhere(r['class_ids'] == 2), axis=0)
    print(ca_indices)
    colours[ca_indices] = [1, 0.5, 0.5]
    colours[jv_indices] = [0.5, 1, 1]
    print(colours)
    #try:
    #    colours[0] = [1, 0.5, 0.5]
    #    colours[1] = [0.5, 1, 1]
    #except:
    #    print("Too few predictions")
    '''
    colours = [[1, 0.5, 0.5],[0.5, 1, 1]]

    print(original_image.shape)
    original_image = np.tile(np.expand_dims(original_image, axis=2), (1, 1, 3))  # Convert grayscale to RBG
    original_image = np.squeeze(original_image, axis=2)

    # Display ground truth on the left
    title = 'GROUND TRUTH: ' + str(idx) + ' / ' + str(len(test_dataset.image_ids) - 1)
    display_instances(original_image, gt_bbox, gt_mask, gt_class_id, test_dataset.class_names,
                      figsize=(8, 8), colors=colours, ax=ax1, title=title)

    # Display prediction on the right
    title = 'PREDICTION: ' + str(idx) + ' / ' + str(len(test_dataset.image_ids) - 1)
    display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                test_dataset.class_names, r['scores'], ax=ax2,
                                title=title, colors=colours)

    for txt in fig.texts:
        txt.remove()
    dice_text = fig.text(0.4, 0.1, "Dice = " + str(dice_score), fontsize='medium')
    if idx in test_indices:
        test_text = fig.text(0.4, 0.0, "IN TEST SET", fontsize='medium')
    fig.canvas.draw()


def predict_image(idx):
    # Make image prediction
    #x = np.squeeze(np.tile(np.expand_dims((X[idx]*255).astype(np.int32), axis=2), (1,1,3)), axis=2)
    x = X[idx]*255
    print("Input shape", x.shape)
    results = model.detect([x], verbose=1)
    r = results[0]
    r = take_best_proposals_only(r)
    # Display prediction on the right
    ax1.clear()
    ax2.clear()
    ax1.imshow(np.squeeze(x,axis=2))
    title = 'PREDICTION: ' + str(idx)
    #print(r)
    x = np.tile(np.expand_dims(x, axis=2), (1, 1, 3))  # Convert grayscale to RBG
    x = np.squeeze(x, axis=2)
    colours = [[1, 0.5, 0.5], [0.5, 1, 1]]
    display_instances(x, r['rois'], r['masks'], r['class_ids'],
                                test_dataset.class_names, r['scores'], ax=ax2,
                                title=title, colors=colours)

def predict_new_image(x):
    results = model.detect([x], verbose=1)
    r = results[0]
    r = take_best_proposals_only(r)
    # Display prediction on the right
    ax1.clear()
    ax2.clear()
    ax1.imshow(np.squeeze(x,axis=2))
    title = 'PREDICTION: '
    #print(r)
    x = np.tile(np.expand_dims(x, axis=2), (1, 1, 3))  # Convert grayscale to RBG
    x = np.squeeze(x, axis=2)
    colours = [[1, 0.5, 0.5], [0.5, 1, 1]]
    display_instances(x, r['rois'], r['masks'], r['class_ids'],
                      test_dataset.class_names, r['scores'], ax=ax2,
                      title=title, colors=colours)


def confidence_interval(n_samples, mean, std_dev, z_value):
    lower = mean - z_value * std_dev / np.sqrt(n_samples)
    upper = mean + z_value * std_dev / np.sqrt(n_samples)
    return (lower, upper)


def evaluate_dataset(dataset):

    # Calculate loss on test set
    test_generator = data_generator(dataset, model.config, shuffle=True)
    #metrics = model.keras_model.evaluate_generator(test_generator, verbose=1)
    #print(metrics)
    #print("test set loss: " + metrics[0])

    # Calculate segmentation-specific metrics
    image_ids = dataset.image_ids
    APs = []
    dice_scores = []
    recalls = []
    specificities = []
    precisions  = []
    for image_id in test_indices:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)
        molded_images = np.expand_dims(mold_image(image, inference_config), 0)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        r = take_best_proposals_only(r)
        #print(r)

        # Compute Average Precision for object detection
        AP = compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], verbose=0)

        # Compute segmentation-specific metrics
        #dice_score, sensitivity, specificity = compute_segmentation_metrics(gt_mask, r['masks'], gt_class_id, r['class_ids'])
        y_true = to_categorical(Y[image_id])
        dice_score = generalized_dice_score(y_true, r['masks'])
        recall, specificity, precision = classification_metrics(y_true, r['masks'])

        print(image_id, ": ", r['masks'].shape, " recall=", recall, " specificity=", specificity, " dice=", dice_score, " precision=", precision)
        APs.append(AP)
        dice_scores.append(dice_score)
        recalls.append(recall)
        specificities.append(specificity)
        precisions.append(precision)

    #print("mean AP (mAP): ", np.mean(APs))
    # Calculate means
    dice_mean = np.mean(dice_scores)
    recall_mean = np.mean(recalls)
    specificity_mean = np.mean(specificities)
    precision_mean = np.mean(precisions)

    # Calculate standard deviations
    dice_std_dev = np.std(dice_scores)
    recall_std_dev = np.std(recalls)
    specificity_std_dev = np.std(specificities)
    precision_std_dev = np.std(precisions)

    # Calculate 95% confidence intervals
    Z_VALUE = 1.96 # From table for Z values
    dice_95_conf = confidence_interval(len(dice_scores), dice_mean, dice_std_dev, Z_VALUE)
    recall_95_conf = confidence_interval(len(recalls), recall_mean, recall_std_dev, Z_VALUE)
    specificity_95_conf = confidence_interval(len(specificities), specificity_mean, specificity_std_dev, Z_VALUE)
    precision_95_conf = confidence_interval(len(precisions), precision_mean, precision_std_dev, Z_VALUE)

    print("Dice: mean = ", dice_mean, " std dev=", dice_std_dev, '95% conf =', dice_95_conf)
    print("Recall/sensitivity: mean = ", recall_mean, " std_dev = ", recall_std_dev, '95% conf =', recall_95_conf)
    print("Specificity: mean = ", specificity_mean, " std dev=", specificity_std_dev, '95% conf =', specificity_95_conf)
    print("Precision: mean = ", precision_mean, " std dev=", precision_std_dev, '95% conf =', precision_95_conf)

# For random test set

test_indices = np.array([ 289,  229, 1401, 1526, 1230,  582,  503,  279,  438, 1504,  135, 1109,  711,  946,
                         1633,  322, 1538, 1234,  161, 1439,  665, 1743,  326,  862, 1406,  259,  376, 1341,
                          390,  539, 1055, 1802, 1326,  384, 1478, 1069,  521,  935, 1302, 1369,  446,  801,
                          688, 1745,  817,  461,  911,  648, 1557,  200,   34, 1078, 1026, 1663, 1669, 1235,
                         1351,  402,  278, 1280, 1083,  564, 1499, 1374,  187, 1440,  517, 1129,  760, 1831,
                           14,  529,  642,  408,  805, 1343,  762, 1185, 1644,  914, 1692, 1145, 1850,  215,
                          393, 1882, 1178,  473,  191, 1573,  963,  220, 1256, 1590,  619,   37, 1378,  769,
                         1416,  981, 1279,  489,  152,  427,  440,  918,  897,  651, 1070, 1830])

# For non-random test set
test_indices = np.arange(start=1901-434, stop=1901)
#test_indices = np.arange(start=0, stop=203)
'''
# For random phantom test set
test_indices = np.array([376, 170, 230, 330, 336, 395, 150,  10,  21, 259, 371,  59, 241, 198, 347,  76, 263, 164,
                         12, 188, 341,  37, 386,  54, 145, 283, 199, 407, 373, 306, 370, 238,  90,  15, 296, 334,
                         298, 141,  74, 272, 356])
'''

orig_shape = (374, 589)
X_path = '/scratch/bvanberl/data/X.npy'
Y_path = '/scratch/bvanberl/data/Y.npy'
#model_path = "/scratch/bvanberl/models/vessels20200211T1602/mask_rcnn_vessels_0100.h5"
model_path = "mask_rcnn.h5"
print("PYTHON VERSION ", sys.version)

# Load the model in inference mode
inference_config = InferenceConfig()
model = MaskRCNN(mode="inference", config=inference_config, model_dir="/scratch/bvanberl/models/vessels20200202T1147/")
#model.compile(inference_config.LEARNING_RATE, inference_config.LEARNING_MOMENTUM)

# Load trained weights
model.load_weights(model_path, by_name=True)

# Create test dataset
test_dataset = VesselDataset(X_path, Y_path)
test_dataset.load_images(0, 1901) # Indices 110-220 are in the test set
test_dataset.prepare()

Y = np.load(Y_path)
X = np.load(X_path)

evaluate_dataset(test_dataset)

