import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

#https://github.com/machrisaa/tensorflow-vgg
import vgg19

#http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
#Above link doesnt work


#np.random.seed(0)
#tf.set_random_seed(0)

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def encoder(X):
    
    with tf.variable_scope("encoder"):
        
        #Store the size of each image
        encoder_layers = {'0': tf.shape(X)}
        
        #Define number of filters
        num_filters = 64
        
        ###########
        #Encoder 1#
        ###########
        Wconv1 = tf.get_variable("Wconv1", shape=[4, 4, 9, num_filters], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        #bconv1 = tf.get_variable("bconv1", shape=[64])
        out = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='SAME') # + bconv1
        
        #Store shape
        encoder_layers['1'] = out
        
        ###########
        #Encoder 2#
        ###########
        
        #Leaky Relu
        out = leaky_relu(out)
        
        #Convolution
        c = int(out.get_shape()[-1])
        Wconv2 = tf.get_variable("Wconv2", shape=[4, 4, c, int(2*c)], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d(out, Wconv2, strides=[1,2,2,1], padding='SAME') # + bconv2 #(Don't Need bias since using batch norm)
        
        #Batch norm
        sconv2 = tf.get_variable("sconv2", shape = [out.get_shape()[-1]] )
        oconv2 = tf.get_variable("oconv2", shape = [out.get_shape()[-1]])
        mc2, vc2 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc2, vc2, oconv2, sconv2, 1e-6)
        
        #Store shape
        encoder_layers['2'] = out
        
        ###########
        #Encoder 3#
        ###########
        
        #Leaky Relu
        out = leaky_relu(out)
        
        #Convolution
        c = int(out.get_shape()[-1])
        Wconv3 = tf.get_variable("Wconv3", shape=[4, 4, c, int(2*c)], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d(out, Wconv3, strides=[1,2,2,1], padding='SAME')
        
        #Batch norm
        sconv3 = tf.get_variable("sconv3", shape = [out.get_shape()[-1]] )
        oconv3 = tf.get_variable("oconv3", shape = [out.get_shape()[-1]])
        mc3, vc3 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc3, vc3, oconv3, sconv3, 1e-6)
        
        #Store shape
        encoder_layers['3'] = out
        
        ###########
        #Encoder 4#
        ###########
        
        #Leaky Relu
        out = leaky_relu(out)
        
        #Convolution
        c = int(out.get_shape()[-1])
        Wconv4 = tf.get_variable("Wconv4", shape=[4, 4, c, int(2*c)], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d(out, Wconv4, strides=[1,2,2,1], padding='SAME')
        
        #Batch norm
        sconv4 = tf.get_variable("sconv4", shape = [out.get_shape()[-1]] )
        oconv4 = tf.get_variable("oconv4", shape = [out.get_shape()[-1]])
        mc4, vc4 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc4, vc4, oconv4, sconv4, 1e-6)
        
        #Store shape
        encoder_layers['4'] = out
        
        ###########
        #Encoder 5#
        ###########
        
        #Leaky Relu
        out = leaky_relu(out)
        
        #Convolution
        c = int(out.get_shape()[-1])
        Wconv5 = tf.get_variable("Wconv5", shape=[4, 4, c, c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d(out, Wconv5, strides=[1,2,2,1], padding='SAME') # + bconv3 #(Don't Need bias since using batch norm)
        
        #Batch norm
        sconv5 = tf.get_variable("sconv5", shape = [out.get_shape()[-1]] )
        oconv5 = tf.get_variable("oconv5", shape = [out.get_shape()[-1]])
        mc5, vc5 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc5, vc5, oconv5, sconv5, 1e-6)
        
        #Store shape
        encoder_layers['5'] = out
        
    return out, encoder_layers


def decoder(out, encoder_layers):
    
    with tf.variable_scope("decoder"):
        
        ###########
        #Decoder 5#
        ###########
        
        #Relu
        out = tf.nn.relu(out)
        
        #Convolution
        b, h, w, c = [int(d) for d in out.get_shape()]
        Wconv5 = tf.get_variable("Wconv5", shape=[4, 4, c, c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d_transpose(out, Wconv5, [b,h*2,w*2,c] ,strides=[1,2,2,1], padding='SAME')
    
        #Batch norm
        sconv5 = tf.get_variable("sconv5", shape = [out.get_shape()[-1]] )
        oconv5 = tf.get_variable("oconv5", shape = [out.get_shape()[-1]])
        mc5, vc5 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc5, vc5, oconv5, sconv5, 1e-6)
        
        #Dropout
        dropout_prob = 0.5
        out = tf.nn.dropout(out, keep_prob=1 - dropout_prob)
        
        ###########
        #Decoder 4#
        ###########
        
        #Concatenate
        e_layer = encoder_layers['4']
        out = tf.concat([out, e_layer], axis=3)
        
        #Relu
        out = tf.nn.relu(out)
        
        #Convolution
        b, h, w, c = [int(d) for d in out.get_shape()]
        Wconv4 = tf.get_variable("Wconv4", shape=[4, 4, int(c/2), c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d_transpose(out, Wconv4, [b,h*2,w*2,int(c/2)] ,strides=[1,2,2,1], padding='SAME')
    
        #Batch norm
        sconv4 = tf.get_variable("sconv4", shape = [out.get_shape()[-1]] )
        oconv4 = tf.get_variable("oconv4", shape = [out.get_shape()[-1]])
        mc4, vc4 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc4, vc4, oconv4, sconv4, 1e-6)
        
        ###########
        #Decoder 3#
        ###########
        
        #Concatenate
        e_layer = encoder_layers['3']
        out = tf.concat([out, e_layer], axis=3)
        
        #Relu
        out = tf.nn.relu(out)
        
        #Convolution
        b, h, w, c = [int(d) for d in out.get_shape()]
        Wconv3 = tf.get_variable("Wconv3", shape=[4, 4, int(c/2), c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d_transpose(out, Wconv3, [b,h*2,w*2,int(c/2)] ,strides=[1,2,2,1], padding='SAME')
    
        #Batch norm
        sconv3 = tf.get_variable("sconv3", shape = [out.get_shape()[-1]] )
        oconv3 = tf.get_variable("oconv3", shape = [out.get_shape()[-1]])
        mc3, vc3 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc3, vc3, oconv3, sconv3, 1e-6)
        
        ###########
        #Decoder 2#
        ###########
        
        #Concatenate
        e_layer = encoder_layers['2']
        out = tf.concat([out, e_layer], axis=3)
        
        #Relu
        out = tf.nn.relu(out)
        
        #Convolution
        b, h, w, c = [int(d) for d in out.get_shape()]
        Wconv2 = tf.get_variable("Wconv2", shape=[4, 4, int(c/2), c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d_transpose(out, Wconv2, [b,h*2,w*2,int(c/2)] ,strides=[1,2,2,1], padding='SAME')
    
        #Batch norm
        sconv2 = tf.get_variable("sconv2", shape = [out.get_shape()[-1]] )
        oconv2 = tf.get_variable("oconv2", shape = [out.get_shape()[-1]])
        mc2, vc2 = tf.nn.moments(out, axes=[0,1,2], keep_dims=False)
        out = tf.nn.batch_normalization(out, mc2, vc2, oconv2, sconv2, 1e-6)
        
        ###########
        #Decoder 1#
        ###########
        
        #Concatenate
        e_layer = encoder_layers['1']
        out = tf.concat([out, e_layer], axis=3)
        
        #Relu
        out = tf.nn.relu(out)
        
        #Convolution
        b, h, w, c = [int(d) for d in out.get_shape()]
        Wconv1 = tf.get_variable("Wconv1", shape=[4, 4, 3, c], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        out = tf.nn.conv2d_transpose(out, Wconv1, [b,h*2,w*2,3] ,strides=[1,2,2,1], padding='SAME')
    
        ##############
        #Final output#
        ##############
        
        #Tanh
        out = tf.sigmoid(out,'out')
        
    return out



def run_model(sess, y_out, mean_loss, X_train, y_train, epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    
    #Shuffle indicies
    train_indicies = np.arange(X_train.shape[0])
    np.random.shuffle(train_indicies)

    #Load saver
    saver = tf.train.Saver()
    
    #Set up variables to compute and optimize. Add training function if training
    fetches = [mean_loss, training]
    
    #Counter 
    batch_counter = 0
    for e in range(epochs):
        
        print('Current epoch: ' + str(e+1))
        
        #Keep track of losses and accuracy
        correct = 0
        losses = []
        
        #Define number of batches
        num_batches = int(X_train.shape[0] / batch_size)
        
        #Iterate through each batch b
        for b in range(num_batches):
            
            #Define current batch
            start_index = b*(batch_size)
            end_index = b*(batch_size)+batch_size
            X_batch = X_train[start_index : end_index]
            y_batch = y_train[start_index : end_index]
        
            #Define the feed dict and run the model
            feed_dict = {X: X_batch,y: y_batch, is_training: True }
            loss, _ = sess.run(fetches,feed_dict=feed_dict)
            
            print('Batch ' + str(b+1) + ' loss = ' + str(loss) )
            
        #Save the model
        saver.save(sess, "model_4/model.ckpt")
        


#Remove the sunglasses from the template
def create_template_mask(I):
    
    #Create a copy of the input
    O = np.ones(I.shape)

    #Loop through each pixel
    i_max = -1
    j_max = -1
    i_min = 1000
    j_min = 1000
    mask = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            
            #If the pixel color matches the color for sunglasses, remove it
            if I[i,j,0] == 170. and I[i,j,1] == 0. and I[i,j,2] == 51.:
                mask.append((i,j))
                if i < i_min:
                    i_min = i
                if i > i_max:
                    i_max = i
                if j < j_min:
                    j_min = j
                if j > j_max:
                    j_max = j
                
                O[i,j,0] = 0
                O[i,j,1] = 0
                O[i,j,2] = 0
    
    #Blur area around glasses
    border_offset = 5
    blur_offset = 10
    for i in range(i_min-border_offset,i_max+border_offset):
        for j in range(j_min - border_offset,j_max + border_offset):
            if (i,j) not in mask:
                for k in range(3):
                    O[i,j,k] = 0
                    #O[i,j,k] = O[i-blur_offset:i+blur_offset,j-blur_offset:j+blur_offset,k].mean()            
    return O




#Initialize X_train
X_train = []
y_train = []

#Load list of training images
template_images = os.listdir('data/template/')
template_images = [i for i in template_images if '.png' in i] #Exclude random files

#Prepare each template image
for t in template_images:
    product_id = t.split('-')[1]
    
    #Skip glasses for now. Only focus on sunglasses
    if int(product_id) <= 555:
        continue
    
    #Get product file name
    front_file = ('data/product/' + product_id + '-front.png')
    angle_file = ('data/product/' + product_id + '-angle.png')
    
    #Load product image arrays and normalize between [0,1]
    front_image = np.asarray(Image.open(front_file)) / 255.
    angle_image = np.asarray(Image.open(angle_file)) / 255.
    front_image.setflags(write=True)
    angle_image.setflags(write=True)

    #Get the template and the parsed images
    template_file = ('data/template/' + t)
    parse_file = ('data/parse/' + t.replace('.png','_vis.png'))
    
    #Load the template and parsed image arrays and normalize between [0,1]
    template_image = np.asarray(Image.open(template_file)) / 255.
    parse_image = np.asarray(Image.open(parse_file))
    
    #Create the modified template file
    template_mask = create_template_mask(parse_image)
    modified_template_image = template_image * template_mask

    #Concatenate the 3 images
    x_input = np.concatenate((modified_template_image, front_image , angle_image), axis=2)
    
    #Add to the training data
    X_train.append(x_input)
    y_train.append(template_image)
    
    #Limit X_train to 320 (multiply of 64)
    if len(X_train) == 320:
        break

#Convert to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

#Extract a sample image
template = y_train[0]
modified_template = X_train[0][:,:,:3]
front = X_train[0][:,:,3:6]
angle = X_train[0][:,:,6:]

#Stats
print('Number of training images: ' + str(len(X_train)) )

#Reset the graph
tf.reset_default_graph()

#Number of training images
m = 64

#Define placeholders
X = tf.placeholder(tf.float32, [m, 224, 224, 9])
y = tf.placeholder(tf.float32, [m, 224, 224, 3])
is_training = tf.placeholder(tf.bool)

#Define model output
y_out_encoder, encoder_history = encoder(X)
y_out = decoder(y_out_encoder, encoder_history)

#Define vgg network
vgg19_y_out = vgg19.Vgg19()
vgg19_y = vgg19.Vgg19()


#Build the vgg network
vgg19_y_out.build(y_out)
vgg19_y.build(y)


#Define loss
g_loss = tf.reduce_mean(tf.abs(y - y_out)) #Loss measured only for the template image


vgg19_y_loss = tf.reduce_mean(tf.abs(vgg19_y.conv1_2 - vgg19_y_out.conv1_2)) + \
                    tf.reduce_mean(tf.abs(vgg19_y.conv2_2 - vgg19_y_out.conv2_2)) + \
                    tf.reduce_mean(tf.abs(vgg19_y.conv3_4 - vgg19_y_out.conv3_4)) + \
                    tf.reduce_mean(tf.abs(vgg19_y.conv4_4 - vgg19_y_out.conv4_4)) + \
                    tf.reduce_mean(tf.abs(vgg19_y.conv5_4 - vgg19_y_out.conv5_4))

                
#Define the loss                
mean_loss = g_loss + vgg19_y_loss

#Define optimizer as an Adam with learning rate of 2e-4 and beta of 0.5
optimizer = tf.train.AdamOptimizer(0.0002, 0.5)

#global_step = tf.get_variable('global_step',0, trainable=False)
train_step = optimizer.minimize(mean_loss)

#Batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

#Load saver
saver = tf.train.Saver()

#Train model
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        
        #Load old model
        #saver.restore(sess, "model_3/model.ckpt")
        
        #Initializer variables
        sess.run(tf.global_variables_initializer())
        
        #Train model
        print('Training')
        run_model(sess, y_out, mean_loss, X_train, y_train, epochs = 200, batch_size = 64, print_every = 100, 
                  training = train_step)
        




