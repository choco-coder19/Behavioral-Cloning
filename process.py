import csv
import cv2
import numpy as np

lines=[]

with open('./Data_buena2/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]
correction=.2

for line in lines:
    for i in range(3):
        source_path=line[i]
        tokens=source_path.split('\\')
        filename=tokens[-1]
        local_path="./Data_buena2/IMG/"+filename
        image=cv2.imread(local_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)


augmented_images=[]
augmented_measurements=[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image=cv2.flip(image,1)
    augmented_images.append(flipped_image)
    augmented_measurements.append(-1*measurement)


X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda , Dropout, ELU
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


model = Sequential()
model.add(Lambda(lambda x: x/255.0 -.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(MaxPooling2D())
model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(ELU())
model.add(Dense(1164))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1))
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,validation_split=.2,shuffle=True,nb_epoch=20)

model.save("model.h5")

