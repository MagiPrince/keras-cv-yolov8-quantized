import hls4ml
import keras_cv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import keras

NAME_BACKBONE = "yolo_v8_xs_backbone"

images_test = np.load("matrices_test.npy")

labels_test = {
    "boxes": np.load("labels_test.npy"),
    "classes": np.load("classes_test.npy")
}

# model = load_model('model.tf', compile=False)
# print(model.get_layer("model_1").get_config())

model = keras_cv.models.YOLOV8DetectorQuantized(
    num_classes=2,
    bounding_box_format="center_xywh",
    backbone=keras_cv.models.YOLOV8BackboneQuantized.from_preset(
        NAME_BACKBONE
    ),
    fpn_depth=2
)

model.build((None, 64, 64, 3))

model.summary()


print("HERE")
print(model.get_config())
print("END")
print("HERE")
print(model.get_layer("model").get_config())
print("END")

# print(model.layers[0].to_json())

# keras.utils.plot_model(model, "my_first_model.png")

# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
# print(model.get_config())
# print(model.get_layer("model"))
config = hls4ml.utils.config_from_keras_model(model.get_config())


print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, output_dir='model_1/hls4ml_prj', part='xcu250-figd2104-2L-e'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()

# # You can print the configuration to see some default parameters
# print(config)

# # Convert it to a hls project
# hls_model = hls4ml.converters.keras_to_hls(config)

# # Print full list of example models if you want to explore more
# hls4ml.utils.fetch_example_list()

# # Use Vivado HLS to synthesize the model
# # This might take several minutes
# hls_model.build()

# # Print out the report if you want
# hls4ml.report.read_vivado_report('my-hls-test')