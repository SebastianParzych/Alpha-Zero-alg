import re
import numpy as np
import keras
import tensorflow as tf

sideCount = 3
maxMoves = sideCount * sideCount

def create_model():
    input = keras.layers.Input(shape = (sideCount, sideCount, 2), dtype = np.float32)
    l2const = 1e-4
    layer = input
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation("relu")(layer)
    for _ in range(8):
        res = layer
        layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.Activation("relu")(layer)
        layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Add()([layer, res])
        layer = keras.layers.Activation("relu")(layer)  
    vhead = layer
    vhead = keras.layers.Dense(36, kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Dense(1)(vhead)
    vhead = keras.layers.Activation("tanh", name = "value_head")(vhead) 
    phead = layer
    phead = keras.layers.Dense(64, kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.BatchNormalization()(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Dense(9)(phead)
    phead = keras.layers.Activation("softmax", name = "policy_head")(phead)  
    model = keras.models.Model(inputs = [input],
                               outputs = [phead, vhead],
                               name='tictactoe_model')
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = [keras.losses.categorical_crossentropy, keras.losses.mean_squared_error],
        loss_weights = [0.5, 0.5],
        metrics=["accuracy"])
    
    
    print(model.summary())
    

    return model