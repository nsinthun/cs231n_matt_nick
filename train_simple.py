from PIL import Image
import os
import numpy as np
import image_utils
import importlib
from keras.layers import Input, Conv2DTranspose, concatenate, Conv2D
from keras.models import Model, Sequential
from keras.backend import variable
from keras import optimizers
importlib.reload(image_utils)
import math
import itertools

front, angle, models = image_utils.load_from_files() #image_utils.load_images(vgg16=True)
orig_front, orig_angle, orig_models = image_utils.load_images()

cutoff = 2
keys = list(models[1].keys())

half_keys = keys[:math.floor(len(keys)/cutoff)]
second_half_keys = keys[math.floor(len(keys)/cutoff):]

test_keys = keys[-3:-1]



##Load in half the data to train and then use the other half to test

#Load in train data
train_x = []
train_y = []
debug_x = []
orig_model_train = []
orig_front_train = []
orig_angle_train = []
debug_y = []

for x in itertools.permutations(half_keys,2):
    orig_prod, new_prod = x
    if 2 in models[1][orig_prod]:
        old_model = models[1][orig_prod][2]
        debug = np.expand_dims(orig_models[1][orig_prod][2], axis=0)
    else:
        old_model = models[1][orig_prod][1]
        debug = np.expand_dims(orig_models[1][orig_prod][1], axis=0)
    next_x_layer = np.concatenate((old_model,front[new_prod],angle[new_prod]),axis=3)
    debug_next_x_layer = np.concatenate((old_model,front[new_prod]),axis=3)
    if 2 in models[1][new_prod]:
        next_y_layer = np.expand_dims(orig_models[1][new_prod][2], axis=0)
    else:
        next_y_layer = np.expand_dims(orig_models[1][new_prod][1], axis=0)
        
    if 2 in models[1][orig_prod]:
        next_orig_mod_layer = np.expand_dims(orig_models[1][orig_prod][2], axis=0)
    else:
        next_orig_mod_layer = np.expand_dims(orig_models[1][orig_prod][1], axis=0)
        
    train_x.append(next_x_layer)
    train_y.append(next_y_layer)
    orig_model_train.append(next_orig_mod_layer)
    orig_front_train.append(np.expand_dims(orig_front[orig_prod],axis=0))
    orig_angle_train.append(np.expand_dims(orig_angle[orig_prod],axis=0))
    debug_y.append(debug)
    debug_x.append(debug_next_x_layer)

train_x = np.concatenate(tuple(train_x),axis=0)
train_y = np.concatenate(tuple(train_y),axis=0)
orig_model_train = np.concatenate(tuple(orig_model_train),axis=0)
orig_front_train = np.concatenate(tuple(orig_front_train),axis=0)
orig_angle_train = np.concatenate(tuple(orig_angle_train),axis=0)
debug_y = np.concatenate(tuple(debug_y),axis=0)
debug_x = np.concatenate(tuple(debug_x),axis=0)


#Load in test data
test_x = []
test_y = []
orig_model_test = []
orig_front_test = []
orig_angle_test = []

for x in itertools.permutations(second_half_keys,2):
    orig_prod, new_prod = x
    if 2 in models[1][orig_prod]:
        old_model = models[1][orig_prod][2]
    else:
        old_model = models[1][orig_prod][1]
    next_x_layer = np.concatenate((old_model,front[new_prod],angle[new_prod]),axis=3)
    if 2 in models[1][new_prod]:
        next_y_layer = np.expand_dims(orig_models[1][new_prod][2], axis=0)
    else:
        next_y_layer = np.expand_dims(orig_models[1][new_prod][1], axis=0)
        
    if 2 in models[1][orig_prod]:
        next_orig_mod_layer = np.expand_dims(orig_models[1][orig_prod][2], axis=0)
    else:
        next_orig_mod_layer = np.expand_dims(orig_models[1][orig_prod][1], axis=0)
        
    test_x.append(next_x_layer)
    test_y.append(next_y_layer)
    orig_model_test.append(next_orig_mod_layer)
    orig_front_test.append(np.expand_dims(orig_front[orig_prod],axis=0))
    orig_angle_test.append(np.expand_dims(orig_angle[orig_prod],axis=0))

test_x = np.concatenate(tuple(test_x),axis=0)
test_y = np.concatenate(tuple(test_y),axis=0)
orig_model_test = np.concatenate(tuple(orig_model_test),axis=0)
orig_front_test = np.concatenate(tuple(orig_front_test),axis=0)
orig_angle_test = np.concatenate(tuple(orig_angle_test),axis=0)



### Build the Model

#Define layers of the model
main_stacked_input = Input(shape=(7, 7, 1024), name='main_stacked_input')
x = Conv2DTranspose(filters=128,kernel_size=2, strides=(2, 2), activation='relu')(main_stacked_input)
x = Conv2DTranspose(filters=64,kernel_size=2, strides=(2, 2), activation='relu')(x)
x = Conv2DTranspose(filters=32,kernel_size=2, strides=(2, 2), activation='relu')(x)
x = Conv2DTranspose(filters=16,kernel_size=2, strides=(2, 2), activation='relu')(x)
int_output = Conv2DTranspose(filters=3,kernel_size=2, strides=(2, 2), activation='tanh')(x)

#Add in original image layers
orig_model_train_tensor = Input(shape=(224, 224, 3), name='orig_model_train_tensor')
orig_front_train_tensor = Input(shape=(224, 224, 3), name='orig_front_train_tensor')
orig_angle_train_tensor = Input(shape=(224, 224, 3), name='orig_angle_train_tensor')
stacked = concatenate(inputs=[int_output,orig_model_train_tensor, orig_front_train_tensor,orig_angle_train_tensor], axis=3)
x = Conv2DTranspose(filters=3,kernel_size=1, strides=(1, 1), activation='tanh')(stacked)
main_output = Conv2D(filters=3,kernel_size=3, padding='same')(x)

#Create and compile model
multi_model = Model(inputs=[main_stacked_input
                            , orig_model_train_tensor, orig_front_train_tensor, orig_angle_train_tensor
                           ], outputs=[main_output])
rms = optimizers.RMSprop(lr=0.005)
multi_model.compile(optimizer= rms, loss='mean_absolute_error')

#Fit the model
multi_model.fit({'main_stacked_input':debug_x, 'orig_model_train_tensor': orig_model_train,
           'orig_front_train_tensor': orig_front_train, 'orig_angle_train_tensor': orig_angle_train
          }, [debug_y], epochs=10, batch_size=32)


#Test the model
stacked_results = multi_model.evaluate({'main_stacked_input':test_x, 'orig_model_train_tensor': orig_model_test,
           'orig_front_train_tensor': orig_front_test, 'orig_angle_train_tensor': orig_angle_test
          }, [test_y])
print(stacked_results)
stacked_predictions = multi_model.predict({'main_stacked_input':test_x, 'orig_model_train_tensor': orig_model_test,
           'orig_front_train_tensor': orig_front_test, 'orig_angle_train_tensor': orig_angle_test
          })
