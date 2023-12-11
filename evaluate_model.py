import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os
import sys
import copy

NAME_BACKBONE = "yolo_v8_xs_backbone"
NAME_WEIGHT = "yolo_v8_xs_backbone"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5


def intersection_over_union(boxA, boxB):
    #Extract bounding boxes coordinates
    x0_A, y0_A, x1_A, y1_A = boxA
    x0_B, y0_B, x1_B, y1_B = boxB

    x1_A, y1_A = x0_A+x1_A, y0_A+y1_A
    x1_B, y1_B = x0_B+x1_B, y0_B+y1_B
    
    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    #Calculate width and height of the intersection area.
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I

    # Handle the negative value width or height of the intersection area
    if width_I < 0 or height_I < 0:
        return 0
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection/union
    # Return the IoU and intersection box
    return IoU

def evaluate_model_f(images_test, labels_test, name_weight, name_backbone):

    model = keras_cv.models.YOLOV8Detector(
        num_classes=2,
        bounding_box_format="center_xywh",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(
            name_backbone
        ),
        fpn_depth=2
    )

    if not os.path.isfile(name_backbone+".h5"):
        print("Aucun fichier de poids trouvé")
        sys.exit(1)

    model.load_weights(name_weight+".h5", skip_mismatch=False, by_name=False, options=None)

    # Get predictions using the model
    results = model.predict(images_test)

    # Confusion Matrix
    true_positif = 0
    false_positif = 0
    false_negative = 0

    nb_gt = 0

    iou_threshold_array = np.arange(start=0.5, stop=1, step=0.05)
    prediction_dict = {}
    for i in range(len(iou_threshold_array)):
        iou_threshold_array[i] = format(iou_threshold_array[i], '.2f')
        prediction_dict[str(iou_threshold_array[i])] = []

    for i in range(len(results["boxes"].to_tensor())):
        detection_over_confidence = 0
        true_detection = 0
        boxes_gt = []
        for j in range(len(labels_test["boxes"][i])):
            if labels_test["classes"][i][j] == 1 and labels_test["boxes"][i][j][0] < 44 and labels_test["boxes"][i][j][0] > 5 and labels_test["boxes"][i][j][1] < 59 and labels_test["boxes"][i][j][1] > 5:
                boxes_gt.append(copy.deepcopy(labels_test["boxes"][i][j].tolist()))

        nb_gt += len(boxes_gt)
        boxes_gt_for_pxrc = {}
        for iou_threshold in iou_threshold_array:
            boxes_gt_for_pxrc[str(iou_threshold)] = copy.deepcopy(boxes_gt)

        for j in range(len(results["boxes"][i])):
            if results["classes"][i][j] == 1 and results["boxes"][i][j][0] < 44 and results["boxes"][i][j][0] > 5 and results["boxes"][i][j][1] > 5 and results["boxes"][i][j][1] < 59:
                index_iou = -1
                best_iou = -1

                pred_box = [tf.get_static_value(results["boxes"][i][j][0]-(results["boxes"][i][j][2]/2)), tf.get_static_value(results["boxes"][i][j][1]-(results["boxes"][i][j][3]/2)), tf.get_static_value(results["boxes"][i][j][2]), tf.get_static_value(results["boxes"][i][j][3])]
                confidence_of_current_box = tf.get_static_value(results["confidence"][i][j])

                if confidence_of_current_box >= CONFIDENCE:
                    detection_over_confidence += 1
                    for k in range(len(boxes_gt)):
                        gt_box = [boxes_gt[k][0]-int(round(boxes_gt[k][2]/2)), boxes_gt[k][1]-int(round(boxes_gt[k][3]/2)), boxes_gt[k][2], boxes_gt[k][3]]
                        iou = intersection_over_union(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            index_iou = k

                    if best_iou >= IOU_THRESHOLD:
                        boxes_gt.pop(index_iou)
                        true_detection += 1
                    

                for iou_threshold in iou_threshold_array:                        
                    # Computing data to generate precision x recall curve
                    index_iou = -1
                    best_iou = -1

                    for k in range(len(boxes_gt_for_pxrc[str(iou_threshold)])):
                        gt_box = [boxes_gt_for_pxrc[str(iou_threshold)][k][0]-int(round(boxes_gt_for_pxrc[str(iou_threshold)][k][2]/2)), boxes_gt_for_pxrc[str(iou_threshold)][k][1]-int(round(boxes_gt_for_pxrc[str(iou_threshold)][k][3]/2)), boxes_gt_for_pxrc[str(iou_threshold)][k][2], boxes_gt_for_pxrc[str(iou_threshold)][k][3]]
                        iou = intersection_over_union(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            index_iou = k

                    if best_iou >= float(iou_threshold):
                        boxes_gt_for_pxrc[str(iou_threshold)].pop(index_iou)
                        prediction_dict[str(iou_threshold)].append((confidence_of_current_box, True))
                    else:
                        prediction_dict[str(iou_threshold)].append((confidence_of_current_box, False))



        true_positif += true_detection
        false_positif += detection_over_confidence-true_detection
        false_negative += len(boxes_gt)

    for iou_threshold in iou_threshold_array:
        prediction_dict[str(iou_threshold)] = sorted(prediction_dict[str(iou_threshold)], key=lambda x: x[0], reverse=True)

    print("True positif : " + str(true_positif))
    print("False positif : " + str(false_positif))
    print("False negative : " + str(false_negative))

    print("F1 score : " + str((2*true_positif)/(2*true_positif+false_positif+false_negative)))

    print("GT : " + str(nb_gt))
    print(len(prediction_dict["0.5"]))

    return true_positif, false_positif, false_negative, ((2*true_positif)/(2*true_positif+false_positif+false_negative)), prediction_dict, nb_gt



def main():
    images_test = np.load("matrices_test.npy")

    print("Nb images : " + str(len(images_test)))

    labels_test = {
        "boxes": np.load("labels_test.npy"),
        "classes": np.load("classes_test.npy")
    }

    true_positif, false_positif, false_negative, f1_score, prediction_array, nb_gt = evaluate_model_f(images_test, labels_test, NAME_WEIGHT, NAME_BACKBONE)
    # print(prediction_array)
    # print(len(prediction_array))
    # print(len([i for i in prediction_array if i[0] >= 0.5]))
    # print(nb_gt)
    # print(true_positif+false_positif)
    print(f1_score)

if __name__ == "__main__":
    main()
