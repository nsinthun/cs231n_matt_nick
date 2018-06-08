from PIL import Image
import os
import numpy as np
from keras.applications.vgg16 import VGG16
import pickle

PRODUCT_PATH = "./data/product"
MODEL_PATH = "./data/model"


def load_images(prod_path=PRODUCT_PATH,model_path=MODEL_PATH,vgg16=False):
    # Returns three dictionaries: 
    #   - the first is a dictionary of the front facing products with the prod number as keys
    #   - the second is the angled products with prod numbers as keys
    #   - the last is a models dict. The model dict is a 3 level dict: the first key is the model number, the second is the product number and the 3rd is the pose number (we have two poses for each)
    if vgg16:
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape = (224,224,3))
        i_p = 0
        i_m = 0
    front = {}
    angle = {}
    models = {}
    for file in os.listdir(PRODUCT_PATH):
        if ".png" in file:
            img = np.array(Image.open(PRODUCT_PATH+"/"+file))
            if vgg16:
                img = vgg16_model.predict(np.expand_dims(img,0))
                i_p += 1
                if(i_p%50 == 0): print("product %i processed" % i_p)
            parts = file.split("-")
            if "front" in parts[1]:
                front[int(parts[0])] = img
            elif "angle" in parts[1]:
                angle[int(parts[0])] = img
            else:
                print("ERROR:" + file)

    for file in os.listdir(MODEL_PATH):
        if ".png" in file:
            img = np.array(Image.open(MODEL_PATH+"/"+file))
            if vgg16:
                img = vgg16_model.predict(np.expand_dims(img,0))
            parts = file.split("model")[1].split("-")
            mod = int(parts[0])
            product = int(parts[1])
            gender = parts[2]
            pose = int(parts[3].split(".")[0])
            if mod not in models:
                models[mod] = {}
            if product not in models[mod]:
                models[mod][product] = {}
            models[mod][product][pose] = img

    return front, angle, models

def load_from_files():
    front = pickle.load(open('front.txt','rb'))
    angle = pickle.load(open('angle.txt','rb'))
    models = pickle.load(open('models.txt','rb'))
    return front, angle, models
