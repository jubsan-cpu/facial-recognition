import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import CosineDecay
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math # Needed for calculating decay steps
from sklearn.metrics import confusion_matrix
import seaborn as sns # For plotting confusion matrix

# --- Explicitly Enable Eager Execution ---
tf.config.run_functions_eagerly(True)
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")
# --- / End Eager Execution ---


# --- Configuration ---
# IMPORTANT: Update this path to where your uploaded dataset folder is located in Colab
dataset_base_dir = '/content/drive/MyDrive/FER2013_7emotions_Uniform_Augmented_Dataset' # <--- UPDATE THIS PATH

train_dir = os.path.join(dataset_base_dir, 'train')
validation_dir = os.path.join(dataset_base_dir, 'validation')
test_dir = os.path.join(dataset_base_dir, 'test')

img_width, img_height = 48, 48
batch_size = 128 # Keeping increased batch size
epochs = 500 # Max epochs
num_classes = 7

# Learning Rate Schedule Config
initial_learning_rate = 0.001
# Calculate total decay steps (total number of training steps over all epochs)
decay_steps = 1 # Placeholder, will calculate after generator is created
alpha = 0.0 # Final learning rate factor (0 means decays to 0)

output_dir = '/content/output_models_cnn_strongaug_cosinedecay' # New output directory
model_save_path = os.path.join(output_dir, 'emotion_model_strongaug_cosinedecay.h5') # New model filename
history_save_path = os.path.join(output_dir, 'training_history_cnn_strongaug_cosinedecay.png')
confusion_matrix_save_path = os.path.join(output_dir, 'confusion_matrix_cnn_strongaug_cosinedecay.png')

# --- Data Loading & STRONGER Augmentation  ---
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.15, zoom_range=0.15, horizontal_flip=True, brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

if not os.path.isdir(train_dir): print(f"Error: Training directory not found at {train_dir}"); import sys; sys.exit(1)
if not os.path.isdir(validation_dir): print(f"Error: Validation directory not found at {validation_dir}"); import sys; sys.exit(1)
if not os.path.isdir(test_dir): print(f"Error: Test directory not found at {test_dir}"); import sys; sys.exit(1)

print("Loading training data with augmentation...")
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size, color_mode='grayscale', class_mode='categorical')
print("Loading validation data...")
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(img_width, img_height), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', shuffle=False)
print("Loading test data...")
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=batch_size, color_mode='grayscale', class_mode='categorical', shuffle=False)

# Calculate decay steps now that we have train_generator.samples
steps_per_epoch_calc = math.ceil(train_generator.samples / batch_size)
decay_steps = steps_per_epoch_calc * epochs # Total steps over max epochs
print(f"Calculated Cosine Decay steps: {decay_steps}")

# Create the Cosine Decay schedule
lr_schedule = CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha
)

# Create the optimizer with the schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


print("Class Indices:", train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())
print("Class Labels (from generator):", class_labels)
actual_labels_generator = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'] # Corrected to lowercase as per common usage in datasets
if sorted(class_labels) != sorted(actual_labels_generator): # Sorted for robust comparison
    print("Warning: Class label order or casing mismatch potentially.")


# --- Model Building ---
def build_cnn_model(input_shape, num_classes, optimizer_to_use):
    # Define the architecture
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.3),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(),
        MaxPooling2D((2, 2)), Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_normal'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the new model with the provided optimizer (which has the schedule)
    model.compile(optimizer=optimizer_to_use,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

input_shape = (img_width, img_height, 1)

print("\n--- Building new model from scratch. ---")
# build a new model
model = build_cnn_model(input_shape, num_classes, optimizer_to_use=optimizer)


# --- Plotting ---
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred_classes, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


# --- Main Execution ---
if __name__ == '__main__':
    start_time = time.time()
    print("\nStarting Colab CNN Training with Stronger Augmentation & Cosine Decay LR...")
    os.makedirs(output_dir, exist_ok=True) # Create the NEW output directory

    # Callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

    print("\nStarting model.fit()...")
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    if train_generator.samples % batch_size != 0: steps_per_epoch += 1
    if validation_generator.samples % batch_size != 0: validation_steps += 1

    initial_epoch = 0

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    print("\nLoading best model weights achieved during this run...")
    if os.path.exists(model_save_path):
        print(f"Reloading best model from {model_save_path} to ensure consistency.")
        try:
            model = load_model(model_save_path) # Load the best saved model, which will be compiled
            print("Best model reloaded for final evaluation.")
        except Exception as e:
            print(f"Error reloading best model: {e}.")
    else:
        print("Warning: Model file not found after training.")

    print("\nEvaluating final model on test set...")
    test_steps = test_generator.samples // batch_size
    if test_generator.samples % batch_size != 0: test_steps += 1
    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    plot_training_history(history, history_save_path)

    print("\nGenerating confusion matrix on test set...")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, steps=test_steps, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels from the test generator
    # Ensure to iterate through the generator to collect all true labels
    y_true = []
    i = 0
    for _ in range(test_steps):
        _, labels_batch = test_generator.next()
        y_true.extend(np.argmax(labels_batch, axis=1))
        i += 1
        if i >= test_steps:
            break
    y_true = np.array(y_true)

    plot_confusion_matrix(y_true, y_pred_classes, class_labels, confusion_matrix_save_path)

    end_time = time.time()
    print(f"\nScript finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Best model saved as '{os.path.basename(model_save_path)}' in '{output_dir}'.")
    print("Please download the .h5 model file if improved.")