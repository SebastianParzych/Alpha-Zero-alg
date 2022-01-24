import re
import numpy as np
import keras
import tensorflow as tf


rowCount, colCount = 6, 7
maxMoves = rowCount * colCount




def create_model():
    input = keras.layers.Input((rowCount, colCount, 2), dtype = np.float32)

    l2const = 1e-4
    layer = input
    layer = keras.layers.ZeroPadding2D((2, 2))(layer)
    layer = keras.layers.Conv2D(96, (4, 4), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.Activation("relu")(layer)
    layer = keras.layers.Conv2D(96, (2, 2), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation("relu")(layer)
    for _ in range(16):
        res = layer
        layer = keras.layers.ZeroPadding2D((2, 2))(layer)
        layer = keras.layers.Conv2D(96, (4, 4), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.Activation("relu")(layer)
        layer = keras.layers.Conv2D(96, (2, 2), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Add()([layer, res])
        layer = keras.layers.Activation("relu")(layer)

    vhead = layer
    vhead = keras.layers.Conv2D(1, (1, 1), kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.BatchNormalization()(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Flatten()(vhead)
    vhead = keras.layers.Dense(32)(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Dense(1)(vhead)
    vhead = keras.layers.Activation("tanh", name = "value_head")(vhead)

    phead = layer
    phead = keras.layers.Conv2D(7, (6, 1), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Conv2D(1, (1, 1), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.BatchNormalization()(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Flatten()(phead)
    phead = keras.layers.Dense(7)(phead)
    phead = keras.layers.Activation("softmax", name = "policy_head")(phead)

    model = keras.models.Model(inputs = [input],
                               outputs = [phead, vhead],
                                name='connect4_model')
    model.compile(
        optimizer = tf.keras.optimizers.Adadelta(),
        loss = [keras.losses.categorical_crossentropy, keras.losses.mean_squared_error],
        loss_weights = [0.5, 0.5],
        metrics=["accuracy"])
    
    return model