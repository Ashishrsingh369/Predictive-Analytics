# Predictive-Analytics
Predictive Maintenance in manufacturing
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(5,)),  # 5 sensor inputs
    layers.Dense(1, activation='sigmoid')  # Failure probability
])
model.compile(optimizer='adam', loss='binary_crossentropy')
# Train with historical sensor data
