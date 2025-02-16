import tensorflow as tf
import numpy as np
from loguru import logger as my_logger
from matplotlib import pyplot as plt

def build_model(input_layer:tf.keras.layers):
    x = tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu")(input_layer)
    x = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    return x

if __name__ == "__main__":
    
    my_logger.info("load mnist dataset")
    mnist_dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    ## look into these images and labels
    # print(f"train_images' shape: {train_images.shape}") # (60000, 28, 28)
    # print(f"train_labels' shape: {train_labels.shape}") # (60000,)
    # plt.figure(); plt.imshow(train_images[0], cmap = "gray") # see image

    my_logger.info("normalize images from 0~255 to 0~1")
    train_images_norm = train_images.astype(np.float32) / 255
    test_images_norm = test_images.astype(np.float32) / 255
    train_images_norm = np.expand_dims(train_images_norm, -1) # (60000, 28, 28, 1)
    test_images_norm = np.expand_dims(test_images_norm, -1)

    my_logger.info("trnasform the labels in one-hot labels") # 5 --> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    num_classes = 10
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes = num_classes)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes = num_classes)

    my_logger.info("build the model")
    input_layer = tf.keras.layers.Input(shape = train_images_norm.shape[1:])
    embedded_network = build_model(input_layer = input_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation = "softmax")(embedded_network)
    my_model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
    my_model.summary() # show model structure

    my_logger.info("Set training params")
    BATCH_SIZE = 128
    EPOCH = 15
    my_model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    my_logger.info("Start training")
    my_model.fit(train_images_norm, train_labels_one_hot, batch_size = BATCH_SIZE, epochs = EPOCH, validation_split = 0.1, shuffle = True)



