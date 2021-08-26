# first neural network with keras tutorial
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers,losses
import string
import re

df=pd.read_csv('./data/tmp.csv')

x=df['text']
y=df['is_offensive']

max_features=20000
embedding_dim=128
sequence_length=500
epochs = 150

vectorize_layer=TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
        )        

#vectorize_layer.adapt(x.batch(64))
def getModel():
    model = tf.keras.models.Sequential()
    model.add(vectorize_layer)
    model.add(layers.Embedding(max_features + 1, embedding_dim))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

'''
    model = tf.keras.models.Sequential()
    model.add(vectorize_layer)
    model.add(layers.Embedding(max_features + 1, embedding_dim))
    model.add(layers.Dropout(0.2))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
'''

model = getModel()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

model.fit(x,y,epochs=epochs)
model.save('./data/nn_model')

_,accuracy=model.evaluate(x,y)

print('Accuracy: %.2f' %(accuracy*100))

print((model.predict(["dog","cat","simp"]) > 0.5).astype("int32"))

