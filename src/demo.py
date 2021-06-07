import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
import os

json_file = open('./Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model_resnet18 = model_from_json(loaded_model_json)
model_resnet18.load_weights("./Model/model_min_val_loss.h5")

for filename in os.listdir('./Dataset/test/'):
    image = cv2.imread('./Dataset/test/' + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image = image / 255
    
    plt.title("Normal" if model_resnet18.predict(image[np.newaxis]) > 0.5 else "Not Normal")
    plt.axis('off')
    plt.imshow(image)
    plt.show()