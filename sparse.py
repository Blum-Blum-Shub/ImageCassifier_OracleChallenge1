import keras.callbacks
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix

import seaborn as sn
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential

from keras import backend as K

if __name__ == "__main__":
    print(tf.config.experimental.list_physical_devices('GPU'))
    print('Reading csv')
    data = pd.read_csv("train.csv")
    idxs = data["idx_train"].tolist()
    paths = data["path_img"].tolist()
    labels = data["label"].tolist()
    images = []

    train_datagen = ImageDataGenerator( validation_split=0.2, horizontal_flip=True) # zoom_range=0.2
    batch_size = 32
    sizeimage = 100
    curated = True
    directory = 'resizedTrain'

    if curated:
        directory += 'Curated'

    if K.image_data_format() == 'channels_first':
        input_shape = (3, sizeimage, sizeimage)
    else:
        input_shape = (sizeimage, sizeimage, 3)

    training_set = train_datagen.flow_from_directory(directory,
                                                     batch_size=batch_size,
                                                     target_size=(sizeimage, sizeimage),
                                                     class_mode='sparse',
                                                     subset='training')
    test_set = train_datagen.flow_from_directory(directory,
                                                 batch_size=batch_size,
                                                 target_size=(sizeimage, sizeimage),
                                                 class_mode='sparse',
                                                 subset='validation')

    confusion = train_datagen.flow_from_directory(directory,
                                                  batch_size=-1,
                                                 target_size=(sizeimage, sizeimage),
                                                 class_mode='sparse',
                                                 subset='validation')

    full_set = train_datagen.flow_from_directory(directory,
                                                     batch_size=batch_size,
                                                     target_size=(sizeimage, sizeimage),
                                                     class_mode='sparse')

    X_train, y_train = training_set.next()
    X_test, y_test = test_set.next()

    X_confusion, y_confusion = confusion.next()

    X_full, y_full = full_set.next()

    '''
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=10, kernel_size=4, activation='relu', input_shape=input_shape))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=4))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=8, activation='relu'))
    '''

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=input_shape),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    cnn = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(8, activation='softmax', name="outputs")
    ])

    #cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    Wsave = cnn.get_weights()
    epochs = 30

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    history = cnn.fit(x=training_set, validation_data=test_set, epochs=epochs, callbacks=[es_callback])

    # Confusion matrix
    y_prediction = cnn.predict(X_confusion)
    y_prediction = np.argmax(y_prediction, axis=1)

    #y_confusion = np.argmax(y_confusion, axis=1)
    result = confusion_matrix(y_confusion, y_prediction, normalize='pred')

    sn.heatmap(result, annot=True, annot_kws={"size": 10})
    plt.show()

    # Train graph
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_epochs = len(val_acc)
    epochs_range = range(final_epochs)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Train with full dataset
    best_epoch = np.argmax(val_acc)
    print(val_acc)
    print("Best epoch is:", best_epoch)

    train_epochs = int(input("For how many epochs do you want to train the finished model: "))
    cnn.set_weights(Wsave)
    print("Starting full sample set fit")
    cnn.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = cnn.fit(x=full_set, epochs=train_epochs)

    model_dir = "saved_models/"
    model_name = "sparse"
    cnn.save(model_dir + model_name)
    print("Model saved as", model_name)