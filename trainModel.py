import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Input)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW # Using AdamW

print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
MODEL_SAVE_PATH = 'emotion_cnn_model_v2.keras' # Changed model name
LABEL_MAP_PATH = 'emotion_labels.npy'
EPOCHS = 60 # Increased epochs, relies on EarlyStopping
BATCH_SIZE = 64 # Can experiment with batch size
IMAGE_HEIGHT, IMAGE_WIDTH = 48, 48 # Should match prepareData.py
NUM_CHANNELS = 1 # Grayscale
L2_REG = 0.001 # L2 regularization factor

# --- Load Data ---
print("Loading preprocessed data...")
try:
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    # Load the label map created by prepareData.py
    emotion_labels_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    num_classes = len(emotion_labels_map)
    emotions = [emotion_labels_map[i] for i in range(num_classes)] # Get names in order
    print(f"Found {num_classes} emotion classes: {emotions}")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please run prepareData.py first.")
    exit()

print("Data shapes:")
print("  X_train:", X_train.shape)
print("  y_train:", y_train.shape)
print("  X_test:", X_test.shape)
print("  y_test:", y_test.shape)

# Ensure data has the channel dimension (should already be there from prepareData)
if len(X_train.shape) == 3:
    print("Adding channel dimension to data...")
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    print("  New X_train shape:", X_train.shape)
    print("  New X_test shape:", X_test.shape)

# --- Preprocessing ---
# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- IMPORTANT: Rescale Test/Validation Data Manually ---
# Since ImageDataGenerator rescales train data, we must do it for test data too.
print("Rescaling test data (dividing by 255.0)...")
X_test_rescaled = X_test / 255.0

# --- Calculate Class Weights ---
print("Calculating class weights...")
class_labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
class_weight_dict = dict(zip(class_labels, class_weights))
print("Class Weights:", class_weight_dict)

# --- Data Augmentation ---
print("Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Apply scaling *only* here for training data
    rotation_range=20,       # Reduced range slightly
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Added brightness augmentation
    fill_mode='nearest'
)

# Create generator for training data
train_generator = train_datagen.flow(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE
)

# --- Build CNN Model ---
print("Building CNN model...")
model = Sequential([
    Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)), # Explicit Input layer

    Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.25), # Optional dropout

    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.3), # Optional dropout

    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.4), # Optional dropout

    # Flatten and Dense Layers
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG)), # Added dense layer
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Output layer
])

# Compile the model
print("Compiling model...")
model.compile(
    optimizer=AdamW(learning_rate=0.001), # Use AdamW with default LR initially
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
print("Setting up callbacks...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
]

# --- Train the Model ---
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test_rescaled, y_test_cat), # Use the rescaled test data for validation
    callbacks=callbacks,
    class_weight=class_weight_dict # Apply class weights
)

print("\nTraining finished.")

# --- Evaluate on Test Set ---
print("\nEvaluating model on test set...")
# Load the *best* saved model for final evaluation
print(f"Loading best model from {MODEL_SAVE_PATH}")
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

loss, acc = best_model.evaluate(X_test_rescaled, y_test_cat, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc * 100:.2f}%")

# --- Detailed Evaluation Metrics ---
print("\nGenerating detailed evaluation report...")
# Get predictions on the (rescaled) test set
y_pred_probs = best_model.predict(X_test_rescaled)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=emotions, digits=3))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_labels)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions, annot_kws={"size": 10})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# Save or show plot
plot_filename = 'confusion_matrix.png'
plt.savefig(plot_filename)
print(f"Confusion matrix plot saved to {plot_filename}")
# To display in Colab:
# plt.show()


# --- Plot Training History ---
print("\nPlotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
axes[0].grid(True)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')
axes[1].grid(True)

plt.tight_layout()
history_plot_filename = 'training_history.png'
plt.savefig(history_plot_filename)
print(f"Training history plot saved to {history_plot_filename}")
# To display in Colab:
# plt.show()

print("\nScript finished.")