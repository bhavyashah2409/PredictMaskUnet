# import os
import cv2 as cv
import numpy as np
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils import TemplateMatcher

IMG_H, IMG_W, IMG_C = 256, 256, 3
BATCH_SIZE = 8

X_inp = tf.keras.Input(shape=(IMG_H, IMG_W, IMG_C), batch_size=BATCH_SIZE)
base = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=X_inp)
s1 = base.get_layer('block1_conv2').output
s2 = base.get_layer('block2_conv2').output
s3 = base.get_layer('block3_conv4').output
s4 = base.get_layer('block4_conv4').output
s5 = base.get_layer('block5_conv4').output
X = base.output
X = tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(X)
X = tf.keras.layers.Concatenate(axis=-1)([X, s5])
X = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(X)
X = tf.keras.layers.Concatenate(axis=-1)([X, s4])
X = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(X)
X = tf.keras.layers.Concatenate(axis=-1)([X, s3])
X = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(X)
X = tf.keras.layers.Concatenate(axis=-1)([X, s2])
X = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(X)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Activation('relu')(X)
X = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(X)
X = tf.keras.layers.Concatenate(axis=-1)([X, s1])
X = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(X)
model = tf.keras.Model(inputs=X_inp, outputs=X)
model.load_weights(r'BhavyaModel\weights.h5')
model.load_weights(r'weights.h5')

def predict_mask(img_path, model):
    fig, ax = plt.subplots(1, 3, figsize=(14, 10))
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (IMG_W, IMG_H))
    ax[0].imshow(img)
    ax[0].axis('off')
    pred = np.array(img, dtype='float32')
    pred = np.expand_dims(pred, axis=0)
    pred = pred / 255.0
    mask = model.predict(pred)
    mask = np.squeeze(mask, axis=0)
    mask = np.round(mask)
    mask = mask * 255.0
    ax[1].imshow(mask)
    ax[1].axis('off')
    mask = np.concatenate([mask, mask, mask], axis=-1)
    maks = 255 - mask
    # print(img.shape, mask.shape)
    img = np.array(img, dtype='uint8')
    mask = np.array(mask, dtype='uint8')
    img = cv.addWeighted(img, 0.5, mask, 0.5, gamma=0)
    ax[2].imshow(img)
    ax[2].axis('off')
    plt.show()

predict_mask(r'anomalib\src\anomalib\models\patchcore\patchcore_data2\oring_absent\89.png', model)
predict_mask(r'anomalib\src\anomalib\models\patchcore\patchcore_data2\oring_present\zpiwnptbgz.png', model)

def predict_mask(img_path, model):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (IMG_W, IMG_H))
    img = np.array(img, dtype='uint8')
    pred = np.array(img, dtype='float32')
    pred = np.expand_dims(pred, axis=0)
    pred = pred / 255.0
    mask = model.predict(pred)
    mask = np.squeeze(mask, axis=0)
    mask = np.round(mask)
    mask = mask * 255.0
    mask = np.concatenate([mask, mask, mask], axis=-1)
    maks = 255 - mask
    mask = np.array(mask, dtype='uint8')
    merged = cv.addWeighted(img, 0.5, mask, 0.5, gamma=0)
    return img, mask, merged

FOLDER = r'anomalib\src\anomalib\models\patchcore\patchcore_data2'
fig, ax = plt.subplots(5, 6, figsize=(20, 15))
for i in range(5):
    for j in range(2):
        if j % 2 == 0:
            img_path = os.path.join(FOLDER, 'oring_present', rn.choice(os.listdir(os.path.join(FOLDER, 'oring_present'))))
        else:
            img_path = os.path.join(FOLDER, 'oring_absent', rn.choice(os.listdir(os.path.join(FOLDER, 'oring_absent'))))
        img, mask, merged = predict_mask(img_path, model)
        ax[i, j * 3 + 0].imshow(img)
        ax[i, j * 3 + 0].set_title(img_path)
        ax[i, j * 3 + 0].axis('off')
        ax[i, j * 3 + 1].imshow(mask)
        ax[i, j * 3 + 1].axis('off')
        ax[i, j * 3 + 2].imshow(merged)
        ax[i, j * 3 + 2].axis('off')
plt.show()

def predict_mask(img, model):
    img = cv.resize(img, (IMG_W, IMG_H))
    pred = np.array(img, dtype='float32')
    pred = np.expand_dims(pred, axis=0)
    pred = pred / 255.0
    mask = model.predict(pred)
    mask = np.squeeze(mask, axis=0)
    mask = np.round(mask)
    mask = mask * 255.0
    mask = np.concatenate([mask, mask, mask], axis=-1)
    maks = 255 - mask
    mask = np.array(mask, dtype='uint8')
    merged = cv.addWeighted(img, 0.5, mask, 0.5, gamma=0)
    return img, mask, merged

VIDEO = r"Nozzle\video10.mp4"
cap = cv.VideoCapture(VIDEO)
matcher = TemplateMatcher(r"template_full.png")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    nozzles = matcher.infer(frame)
    # print(nozzles)
    if len(nozzles) > 0:
        nozzle = nozzles[0][0]
        nozzle, mask, merged = predict_mask(nozzle, model)
        cv.imshow('Nozzle', nozzle)
        cv.imshow('Mask', mask)
        cv.imshow('Merged', merged)
    else:
        if cv.getWindowProperty('Nozzle', cv.WND_PROP_VISIBLE):
            cv.destroyWindow('Nozzle')
        if cv.getWindowProperty('Mask', cv.WND_PROP_VISIBLE):
            cv.destroyWindow('Mask')
        if cv.getWindowProperty('Merged', cv.WND_PROP_VISIBLE):
            cv.destroyWindow('Merged')
    # merged = cv.addWeighted(frame, 0.5, mask, 0.5, gamma=0)
    frame = cv.resize(frame, None, None, fx=0.5, fy=0.5)
    cv.imshow('Frame', frame)
    # cv.imshow('Merged', merged)
    if cv.waitKey(1) == 27:
        break
cap.release()

import cv2 as cv
from Utils import TemplateMatcher

VIDEO = r"Nozzle\video3.mp4"
cap = cv.VideoCapture(VIDEO)
fps = cap.get(cv.CAP_PROP_FPS)
w, h = int(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) / 2), int(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) / 2)
writer = cv.VideoWriter('result_yes.mp4', cv.VideoWriter_fourcc(*"MPEG"), fps, (w, h))
matcher = TemplateMatcher(r"template_full.png", 0.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    match = matcher.infer(frame)
    if len(match) > 0:
        xmin, ymin, xmax, ymax = match[0][-1]
        frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        frame = cv.putText(frame, 'ORING: YES', (10, 150), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
    else:
        frame = cv.putText(frame, 'ORING:', (10, 150), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    frame = cv.resize(frame, None, None, fx=0.5, fy=0.5)
    cv.imshow('Frame', frame)
    writer.write(frame)
    if cv.waitKey(1) == 27:
        break
writer.release()
cap.release()
