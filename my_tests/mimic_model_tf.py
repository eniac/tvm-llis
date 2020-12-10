import tvm
from tvm import te
import tvm.relay as relay

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()

model.add(layers.LSTM(180, stateful=True, return_sequences=True, time_major=True, input_shape=(1, 18), dtype='float32'))
model.add(layers.LSTM(180, stateful=True, time_major=True))
model.add(layers.Dense(2))

#model = tf.python.keras.engine.functional.Functional(model)

print(model.summary())

#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#keras_resnet50 = keras.applications.resnet50.ResNet50(
#    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
#)

shape_dict = {"input_1": (1, 1, 18)}
mod, params = relay.frontend.from_keras(model, shape_dict)
target = "cuda_kelvin"
target_host = "c"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)
    lib.export_library("mimic-keras-pack.so")

