import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

NAME_BACKBONE = "yolo_v8_xs_backbone"
NAME_WEIGHT = "yolo_v8_xs_backbone_tmp"
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

model = keras_cv.models.YOLOV8Detector(
    num_classes=2,
    bounding_box_format="center_xywh",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        NAME_BACKBONE
    ),
    fpn_depth=2
)

model.summary()

# Evaluate model
# model(images)

if os.path.isfile(NAME_BACKBONE+".h5") and not TRAIN:
    model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

else:
    # model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)
    # Train model
    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='iou',
        optimizer=tf.optimizers.Adam(),
        jit_compile=False,
    )

    earlyStopping = EarlyStopping(monitor='loss', patience=400, verbose=0, mode='min')
    mcp_save_val_loss_min = ModelCheckpoint('val_loss_min.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
    mcp_save_loss_min = ModelCheckpoint('loss_min.h5', save_best_only=True, save_weights_only=True, monitor='loss', mode='min')
    # mcp_save_val_accuracy_max = ModelCheckpoint('val_accuracy_max.h5', save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max')
    # mcp_save_accuracy_max = ModelCheckpoint('accuracy_max.h5', save_best_only=True, save_weights_only=True, monitor='accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=500, verbose=1, mode='min')

    model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=1, batch_size=64, callbacks=[mcp_save_val_loss_min, mcp_save_loss_min])

    model.save_weights(NAME_WEIGHT+".h5", overwrite="True", save_format="h5", options=None)

    # acc = model.history.history['loss']
    # print(acc) # [0.9573, 0.9696, 0.9754, 0.9762, 0.9784]


# model.summary()
# Get predictions using the model
# print(model.predict(images_test[:1]))
# print("Boxes : ", labels_test["boxes"][:1])
# print("Classes : ", labels_test["classes"][:1])

# tf.keras.saving.save_model(model, "model.tf")
