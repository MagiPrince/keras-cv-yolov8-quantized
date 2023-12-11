import keras_cv
import tensorflow as tf
import numpy as np
import os
import sys
import time

NAME_BACKBONE = "yolo_v8_xs_backbone"
NAME_WEIGHT = "yolo_v8_xs_backbone_tmp"

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

# model.summary()

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
######################################Calculate FLOPS##########################################
def get_flops(model):
    '''
    Calculate FLOPS
    Parameters
    ----------
    model : tf.keras.Model
        Model for calculating FLOPS.

    Returns
    -------
    flops.total_float_ops : int
        Calculated FLOPS for the model
    '''
    
    batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

flops = get_flops(model)
print(f"FLOPS: {flops}")




images_test = np.load("matrices_test.npy")

print("Nb images : " + str(len(images_test)))

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

if not os.path.isfile(NAME_BACKBONE+".h5"):
    print("Aucun fichier de poids trouvé")
    sys.exit(1)

model.load_weights(NAME_WEIGHT+".h5", skip_mismatch=False, by_name=False, options=None)

array_timings = np.zeros(500)

for i in range(500):
    start_time = time.time()
    model.predict(images_test)
    end_time=time.time()
    array_timings[i] = end_time - start_time


print("Nb images : " + str(len(images_test)))
print("Average inference time : " + str(array_timings.mean()))
print("Time for one detection : " + str(array_timings.mean()/len(images_test)))
print("Time for one detection in ms : " + str((array_timings.mean()/len(images_test))*1000))
print("FPS : " + str(1/(array_timings.mean()/len(images_test))))