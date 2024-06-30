import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

# Define input layers for each branch
input_pool1 = Input(shape=(109, 109, 32))  # Shape for pool1 branch
input_dense1 = Input(shape=(64,))          # Shape for dense1 branch

# Define the pool1 branch
pool1_flat = Flatten()(input_pool1)        # Flatten the 4D tensor to 2D

# Concatenate the branches
concatenated = Concatenate(axis=-1)([pool1_flat, input_dense1])

# Add additional layers if needed
combined = Dense(128, activation='relu')(concatenated)
output = Dense(10, activation='softmax')(combined)  # Example output layer with 10 classes

# Create the model
model = Model(inputs=[input_pool1, input_dense1], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Example dummy data to test the model fitting
import numpy as np

# Create dummy data matching the input shapes
dummy_pool1 = np.random.random((10, 109, 109, 32)).astype(np.float32)
dummy_dense1 = np.random.random((10, 64)).astype(np.float32)
dummy_labels = np.random.randint(0, 10, size=(10, 1))
dummy_labels = tf.keras.utils.to_categorical(dummy_labels, num_classes=10)

# Fit the model with dummy data
model.fit([dummy_pool1, dummy_dense1], dummy_labels, epochs=1)
