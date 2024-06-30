from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import pre_processing

X_train, y_train, X_val, y_val, X_test, y_test, train_images, val_images, test_images = pre_processing.preprocessing()

# Load the saved model
model = load_model('my_model.keras')

def create_confusion_matrix(y_test, y_pred):

    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test_labels, y_pred_labels)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")
    plt.close()

if __name__ == '__main__':
    y_pred = model.predict(test_images)
    create_confusion_matrix(y_test, y_pred)