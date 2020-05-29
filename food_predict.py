
import numpy as np
import operator
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image
import cv2
#load model
img_width, img_height = 128, 128
model_path = 'model.h5'
model_weights_path = 'weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

#Prediction on a new picture
from keras.preprocessing import image as image_utils

from PIL import Image, ImageTk
import requests
def Predict():
    class_labels = ['1', '2', '3', '4']
    test_image = image.load_img('crop.jpeg', target_size = (128, 128))
    #test_image = cv2.resize(test, (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    test_image /= 255
    result = model.predict(test_image)

    decoded_predictions = dict(zip(class_labels, result[0]))

    decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)
    #print(decoded_predictions[0][0])
    return decoded_predictions[0][0]
#t_image = image.load_img('3.jfif')



