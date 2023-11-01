import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from evaluate_model import evaluate_model_f

NAME_BACKBONE = "yolo_v8_xs_backbone"
TRAIN = True

images = np.load("matrices_training.npy")

labels = {
    "boxes": np.load("labels_training.npy"),
    "classes": np.load("classes_training.npy")
}

images_validation = np.load("matrices_validation.npy")

labels_validation = {
    "boxes": np.load("labels_validation.npy"),
    "classes": np.load("classes_validation.npy")
}

images_test = np.load("matrices_test.npy")

labels_test = {
    "boxes": np.load("labels_test.npy"),
    "classes": np.load("classes_test.npy")
}

array_epochs = [2, 5, 10, 20, 30, 50, 100, 300, 500]
array_epochs_b4_evaluation = [2, 3, 5, 10, 10, 20, 50, 200, 200]
array_name_weights = ["yolo_v8_xs_backbone"]

dict_results = {}

# Loop of the number of models we are going to train to get training and results data
for iteration in range(0, 5):
    print("--------------------- Experience " + str(iteration) + " ---------------------")
    # Creating model for this experience
    tf.keras.utils.set_random_seed((iteration+1)*17*100)
    model = keras_cv.models.YOLOV8Detector(
        num_classes=2,
        bounding_box_format="center_xywh",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(
            NAME_BACKBONE
        ),
        fpn_depth=2
    )

    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='iou',
        optimizer=tf.optimizers.Adam(),
        jit_compile=False,
    )
    # Setting or creating variables for this experience
    dict_results[str(iteration)] = {}
    loss_history = []
    val_loss_history = []
    for i, nb_epochs in enumerate(array_epochs_b4_evaluation):
        print("--------------------- Working on epochs " + str(array_epochs[i]) + " ---------------------")
        dict_results[str(iteration)][str(array_epochs[i])] = {}

        if i != 0:
            model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

        model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=nb_epochs, batch_size=64)

        model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)

        for j, name_weights in enumerate(array_name_weights):

            true_positif, false_positif, false_negative, f1_score, prediction_array, nb_gt = evaluate_model_f(images_test, labels_test, name_weights, NAME_BACKBONE)
            dict_results[str(iteration)][str(array_epochs[i])]["values_computed"] = [true_positif, false_positif, false_negative, f1_score, nb_gt]
            dict_results[str(iteration)][str(array_epochs[i])]["prediction"] = prediction_array


        loss_history += model.history.history['loss']
        val_loss_history += model.history.history['val_loss']
        

    dict_results[str(iteration)]['loss'] = loss_history
    dict_results[str(iteration)]['val_loss'] = val_loss_history

np.save('YOLOv8_data_generated_backbone_xs_0_5.npy', dict_results)

# acc = model.history.history['val_accuracy']
# print(acc) # [0.9573, 0.9696, 0.9754, 0.9762, 0.9784]

