# -*- coding: utf-8 -*-

from keras.preprocessing import image #This package is used to load or process an image
from keras.models import load_model # This package is used to load model
import numpy as np


image_width=100
image_height=100

img=image.load_img('dataset/training_set/cats/cat.40.jpg',target_size=(image_width,image_height))
print(img)
img=image.img_to_array(img) #convert image into (height,width,channels)
print(img)
img=np.expand_dims(img, axis=0) # convert matrix into (1,height,width,channels)
print(img)
img/=255 # img is np array
print(img)


model=load_model('model.h5')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classes=model.predict_classes(img)
for x in classes:
    if x[0]==0:
        print('dog')
    else:
        print('cat')
#print(classes)
