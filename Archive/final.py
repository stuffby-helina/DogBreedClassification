import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import (
    Dense,
    MaxPooling2D,
    Dropout,
    Flatten,
    BatchNormalization,
    Conv2D,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str, default="")
    return parser.parse_args()


# !pip install kagglehub
# This one takes 3 minutes
import kagglehub


def main():
    args = parse_args()
    data = args.data

    # Download latest version
    path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

    print("Path to dataset files:", path)

    # Loading data into the memory is not efficient since it doesn't have enough memory for that much data
    # Load filenames into dataframe instead

    images = []

    labels = []

    label_count = 0

    images_path = path + r"/images/Images/"

    for label in os.listdir(images_path):
        label_path = images_path + label + "/"
        label_count += 1
        for file in os.listdir(label_path):
            images.append(label_path + file)
            labels.append(label.split("-")[1])

    df = pd.DataFrame({"image_path": images, "label": labels})

    # df = df[:738]

    X_train, X_temp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    label_test_val = X_temp["label"]

    # 10%.   10%
    X_test, X_val = train_test_split(
        X_temp, test_size=0.5, stratify=label_test_val, random_state=42
    )

    print("The shape of train data", X_train.shape)
    print("The shape of test data", X_test.shape)
    print("The shape of validation data", X_val.shape)

    # parameters
    image_size = 255  # Size of the image
    image_channel = 3  # Colour scale (RGB)
    bat_size = 1  # Number of files/images processed at once

    # Applyingimage data gernerator to train and test data
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1.0 / 255,  # to bring the image range from 0..255 to 0..1
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0,  # randomly zoom image
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,
    )  # randomly flip images

    train_generator = datagen.flow_from_dataframe(
        X_train,
        x_col="image_path",
        y_col="label",
        batch_size=bat_size,
        target_size=(image_size, image_size),
        class_mode="categorical",
    )
    val_generator = datagen.flow_from_dataframe(
        X_val,
        x_col="image_path",
        y_col="label",
        batch_size=bat_size,
        target_size=(image_size, image_size),
        shuffle=False,
        class_mode="categorical",
    )

    test_generator = datagen.flow_from_dataframe(
        X_test,
        x_col="image_path",
        y_col="label",
        batch_size=bat_size,
        target_size=(image_size, image_size),
        shuffle=False,
        class_mode="categorical",
    )

    # use the data generator
    num_classes = len(test_generator.class_indices)

    plt.figure(figsize=(10, 8))
    # Plots our figures
    for i in range(1, 5):
        plt.subplot(1, 4, i)
        batch = next(test_generator)
        image_ = batch[0][0]
        plt.imshow(image_)
    plt.show()

    # Load the VGG16 model
    base_model = VGG16(
        weights="imagenet",
        input_shape=(image_size, image_size, image_channel),
        include_top=False,
    )

    # freezing the base model
    base_model.trainable = False

    model = Sequential()

    # Add maxpooling layers along with convolutional layers
    # change activation funcitnos to leaky_relu
    # add a dense layer after the flatten

    model.add(base_model)

    # Input Layer
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="leaky_relu",
            input_shape=(image_size, image_size, image_channel),
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(filters=64, kernel_size=2, activation='leaky_relu'))
    model.add(Dropout(0.2))

    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=128 , kernel_size=2 , padding='same' , activation='leaky_relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(filters=256 , kernel_size=2 , padding='same' , activation='leaky_relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.05)))
    model.add(Dropout(0.4))

    # Output layer
    model.add(
        Dense(num_classes, activation="softmax")
    )  # Softmax for binary classification

    model.summary()

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    fitted = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
    )

    history = model.history
    save_loss_plot(history.history, args, 1)
    # Fine-tuning: Unfreeze some layers in ResNet
    for layer in base_model.layers[-5:]:  # Unfreeze the last 5 layers
        layer.trainable = True

    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fine-tune the model
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
    )

    save_loss_plot(history_fine.history, args, 1)


def save_loss_plot(history, args, num):
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("model accuracy")
    plt.legend()
    plt.savefig(args.main_dir + f"/graph{num}.png")
    print(args.main_dir + f"/graph{num}.png")


if __name__ == "__main__":
    main()
