import os

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

from pre_processing import preprocessing

X_train, y_train, X_val, y_val, X_test, y_test, train_images, val_images, test_images = preprocessing()

print(X_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

image_width = image_height = 220
channels = 1
batch_size = 16
epochs = 50  # Increase epochs for better training
num_classes = 15

# Create the model
model = Sequential([
    Input(shape=(image_width, image_height, channels)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),  # Adjust learning rate
              loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model with validation
history = model.fit(
    train_images, y_train,
    validation_data=(val_images, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

# Save the entire model to a HDF5 file
model.save('my_model.keras')

import confusion_matrix

# Get predictions
y_pred = model.predict(val_images)

# Compute confusion matrix
#cm = confusion_matrix.create_confusion_matrix(y_test, y_pred)