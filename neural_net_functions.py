import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tensorflow as tf
import keras
from keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
from functions import nn_regression_preprocess


def neural_net(path):
    dataframe = pd.read_csv(path)
    dataset = nn_regression_preprocess(dataframe)
    target = 'impervious_1'
    features = [
        'landsat_1', 'landsat_2', 'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6',  
        'aspect_1_0', 'aspect_1_1', 'aspect_1_2', 'aspect_1_3', 'aspect_1_4',
        'aspect_1_5', 'aspect_1_6', 'aspect_1_7', 'aspect_1_8', 'aspect_1_9', 
        'aspect_1_10', 'aspect_1_11', 'aspect_1_12', 'aspect_1_13', 'aspect_1_14',
        'aspect_1_15', 'aspect_1_16', 'aspect_1_17','aspect_1_18', 
        'wetlands_1_0', 'wetlands_1_2', 'wetlands_1_3', 'wetlands_1_4', 
        'wetlands_1_5', 'wetlands_1_6', 'wetlands_1_7', 'wetlands_1_8',
        'dem_1', 'posidex_1', 'NDVI'
    ]
    X = dataset[features]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = DNN_functional(36, 1, 'linear')
    history = run_training(model, (X_train, y_train), (X_test, y_test), 10, 1024, 1e-4, 'adam', 'mse', 'mae')
    model = tf.keras.models.load_model("model.h5", compile=False)

def get_compiled_model(model, lr, opt, loss, metric):
    print("Compiling and returning model")
    model.summary()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[metric]
        )
    return model


def dense_block(x, nlayers, dropout=0, batch_norm=False):
    x = tf.keras.layers.Dense(nlayers, activation='relu')(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    return x

def DNN_functional(shape, out_dim, activ, norm=False):
    inputs = tf.keras.layers.Input(shape) 
    x = dense_block(inputs, 64, 0, norm)
    x = dense_block(x, 128, 0, norm)
    x = dense_block(x, 1024, 0.5, norm)
    x = dense_block(x, 256, 0.2, norm)
    output = tf.keras.layers.Dense(out_dim, activation=activ)(x)
    return tf.keras.Model(inputs=inputs, outputs=output, name='DNN')

def run_training(model, tdataset, vdataset, 
                 epochs, batch_size, lr, opt, loss, metric):
    
    model = get_compiled_model(model, lr, opt, loss, metric)
    print("Setting Callbacks")
    model_output = tf.keras.callbacks.ModelCheckpoint(f"model.h5", monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)    
    print(f"Training model")
    return model.fit(
        tdataset[0],
        tdataset[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(vdataset[0], vdataset[1]),
        callbacks=[model_output]
        )

