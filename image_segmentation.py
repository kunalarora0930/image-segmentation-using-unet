
import opendatasets as od
import pandas as pd

od.download('https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset/') # insert ypu kaggle  username and key

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

images_path = "/content/segmentation-full-body-tiktok-dancing-dataset/segmentation_full_body_tik_tok_2615_img/segmentation_full_body_tik_tok_2615_img/images"
masks_path = "/content/segmentation-full-body-tiktok-dancing-dataset/segmentation_full_body_tik_tok_2615_img/segmentation_full_body_tik_tok_2615_img/masks"

file_names = os.listdir(images_path)
file_names[:5]

def load_image(image_path, mask_path):
    image, mask = tf.io.read_file(image_path), tf.io.read_file(mask_path)
    image, mask = tf.image.decode_png(image), tf.image.decode_png(mask)
    image, mask = tf.cast(image, tf.float32), tf.cast(mask, tf.float32)
    image, mask = tf.image.resize(image, (224,224)), tf.image.resize(mask, (224,224))
    image, mask = image/255, mask/255
    return image, mask

image, mask = load_image(os.path.join(images_path, file_names[0]), os.path.join(masks_path, file_names[0]))
image, mask

fig, axes = plt.subplots(1,2,figsize=(9,9))
axes[0].imshow(image)
axes[1].imshow(mask)

import keras
from keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model

inputs = Input((224,224,3))

c1 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (inputs)
c1 = BatchNormalization()(c1)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (c1)
c1 = BatchNormalization()(c1)
c1 = Dropout(0.1)(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (p1)
c2 = BatchNormalization()(c2)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (p2)
c3 = BatchNormalization()(c3)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (p3)
c4 = BatchNormalization()(c4)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(512, (3,3), activation="relu", kernel_initializer="he_normal", padding="same") (p4)
c5 = BatchNormalization()(c5)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(0.2)(c8)
c8 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
c9 = BatchNormalization()(c9)
c9 = Dropout(0.2)(c9)
c9 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)
c9 = BatchNormalization()(c9)

outputs=Conv2D(1, (1,1), activation="sigmoid")(c9)



unet_model = Model(inputs=[inputs], outputs=[outputs])
unet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
unet_model.summary()

images = np.array([os.path.join(images_path, x) for x in file_names])
masks = np.array([os.path.join(masks_path, x) for x in file_names])
print(images.shape, masks.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(images, masks, test_size=0.2)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

training_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
training_dataset = training_dataset.map(load_image).shuffle(1000).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.map(load_image).batch(1)

loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x,y):
    loss_ = 0
    with tf.GradientTape(persistent=True) as tape:
        pred = unet_model(x, training=True)
        loss_ = loss(pred,y)
    gradients = tape.gradient(loss_, unet_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, unet_model.trainable_variables))
    return loss_

epochs = 100
losses = []
for epoch in range(epochs):
    batch_loss = []
    for train_x, train_y in training_dataset:
        loss_ = train_step(train_x, train_y)
        batch_loss.append(loss_)
    val_losses = []
    for x,y in test_dataset:
        val_losses.append(loss(unet_model(x, training=False), y))
    losses.append(np.mean(batch_loss))
    print(f"Completed epoch: {epoch} Training loss: {np.mean(batch_loss)} Validation loss: {np.mean(val_losses)}")

x,y = next(iter(test_dataset))
x.shape, y.shape

pred = unet_model(x)
pred.shape

plt.imshow(pred[0])

plt.imshow(y[0])

unet_model.save("/kaggle/working/model")