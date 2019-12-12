import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.regularizers import l2
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def cosine_distance(vecs,normalize=False):
    x,y = vecs
    if normalize:
        x = K.l2_normalize(x,axis=0)
        y = K.l2_normalize(y,axis=0)
    return K.prod(K.stack([x,y],axis=1),axis=1)
def cosine_distance_output_shape(shapes):
    return shapes[0]
def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

def my_init_bias(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)
def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(32, kernel_size =(3,3), activation='relu', input_shape=input_shape,
                   kernel_initializer=my_init, kernel_regularizer=l2(2e-4)))
   
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64,  kernel_size =(3,3), activation='relu',
                     kernel_initializer=my_init,
                     bias_initializer=my_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size = (3,3), activation='relu', kernel_initializer=my_init,
                     bias_initializer=my_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
#    model.add(Conv2D(128, kernel_size = (3,3), activation='relu', kernel_initializer=my_init,
#                     bias_initializer=my_init_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=my_init,bias_initializer=my_init_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    #L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #L1_distance = L1_layer([encoded_l, encoded_r])
    L1_distance = Lambda(cosine_distance, output_shape=cosine_distance_output_shape)([encoded_l,encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=my_init_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


def gen_random_batch(in_groups, batch_halfsize = 8):
    out_img_a, out_img_b, out_score = [], [], []
    all_groups = list(range(len(in_groups)))
    for match_group in [True, False]:
        group_idx = np.random.choice(all_groups, size = batch_halfsize)
        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*batch_halfsize
        else:
            # anything but the same group
            #non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 
            #b_group_idx = non_group_idx
            out_score += [0]*batch_halfsize
            
        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
            
    return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)


img_folder = './victim-set/'
train_images = []
train_responses = [13, 13, 13, 13, 13, 13]
fig = plt.figure(figsize=(16,4))
fig.suptitle('Train images from victim-set', fontsize=16)

cnt = 0
for img_name in os.listdir(img_folder):
    if not img_name.endswith('.ipynb_checkpoints'):
        image = mpimg.imread(img_folder+img_name)
        image_to_show = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
        cnt = cnt+1
        #sub=plt.subplot(1,5,cnt)
        #sub.set_title(img_name)
        #plt.imshow(image_to_show)
        filename = str(cnt)+".png"
        #figure = plt.figure()
        #plt.imsave(filename,image_to_show)
        #image = preprocess_image(image) 
        train_images.append(image)

"""        
train_images = np.array(train_images)
print(train_images.shape)
n_classes = 1
y = np.zeros([train_images.shape[0],n_classes])
y[:,0] = 1
print(y[0])
X_full = train_images
y_full = y
        
x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3)
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
y_train = y_train.astype('int')
y_test = y_test.astype('int')
print('Training', x_train.shape, x_train.max())
print('Testing', x_test.shape, x_test.max())
"""

        
perturbed_image_folder = './optimization_output/l1basedmask_uniformrectangles/noisy_images/'


train_images_2 = []
train_responses_2 = [13, 13, 13, 13, 13, 13]
fig = plt.figure(figsize=(16,4))
fig.suptitle('Train images from noisy_images', fontsize=16)

cnt = 0
for img_name in os.listdir(perturbed_image_folder):
    if not img_name.endswith('.ipynb_checkpoints'):
        image = mpimg.imread(perturbed_image_folder+img_name)
        image_to_show = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
        cnt = cnt+1
        #sub=plt.subplot(1,5,cnt)
        #sub.set_title(img_name)
        #plt.imshow(image_to_show)
        filename = str(cnt)+".png"
        #figure = plt.figure()
        #plt.imsave(filename,image_to_show)
        #image = preprocess_image(image) 
        #train_images_2.append(image)
        train_images.append(image)

        
experimental_image_folder = './experimental_attack_images-edited/' 


cnt = 0
for img_name in os.listdir(experimental_image_folder):
    if not img_name.endswith('.ipynb_checkpoints'):
        image = mpimg.imread(experimental_image_folder+img_name)
        image_to_show = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
        cnt = cnt+1
        #sub=plt.subplot(1,5,cnt)
        #sub.set_title(img_name)
        #plt.imshow(image_to_show)
        filename = str(cnt)+".png"
        #figure = plt.figure()
        #plt.imsave(filename,image_to_show)
        #image = preprocess_image(image) 
        #train_images_2.append(image)
        train_images.append(image)
#train_images_2 = np.array(train_images_2)
#print(train_images_2.shape)
train_images = np.array(train_images)
print(train_images.shape)
n_classes = 1
#y = np.zeros([train_images_2.shape[0],n_classes])
#y[:,0] = 1
#print(y[0])
#X_full_2 = train_images_2
#y_full_2 = y
y = np.zeros([train_images.shape[0],n_classes])
y[:,0] = 1
print(y[0])
X_full_2 = train_images
y_full_2 = y
        
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X_full_2, y_full_2, test_size = 0.3)
x_train_2 = x_train_2.reshape(-1, 32, 32, 3).astype('float32') / 255.
x_test_2 = x_test_2.reshape(-1, 32, 32, 3).astype('float32') / 255.
y_train_2 = y_train_2.astype('int')
y_test_2 = y_test_2.astype('int')
print('Training', x_train_2.shape, x_train_2.max())
print('Testing', x_test_2.shape, x_test_2.max())

img_in = Input(shape = x_train_2.shape[1:], name = 'FeatureNet_ImageInput')
n_layer = img_in
for i in range(2):
    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    n_layer = MaxPool2D((2,2))(n_layer)
n_layer = Flatten()(n_layer)
n_layer = Dense(32, activation = 'linear')(n_layer)
n_layer = Dropout(0.5)(n_layer)
n_layer = BatchNormalization()(n_layer)
n_layer = Activation('relu')(n_layer)
feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')
feature_model.summary()

train_groups = [x_train_2[np.where(y_train_2==i)[0]] for i in np.unique(y_train_2)]
test_groups = [x_test_2[np.where(y_test_2==i)[0]] for i in np.unique(y_train_2)]

print('train groups:', [x.shape[0] for x in train_groups])
print('test groups:',[x.shape[0] for x in test_groups])

img_a_in = Input(shape = x_train_2.shape[1:], name = 'ImageA_Input')
#img_b_in = Input(shape = x_train.shape[1:], name = 'ImageB_Input')
img_b_in = Input(shape = x_train_2.shape[1:], name = 'ImageB_Input')
img_a_feat = feature_model(img_a_in)
img_b_feat = feature_model(img_b_in)
combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
combined_features = Dense(16, activation = 'linear')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Activation('relu')(combined_features)
combined_features = Dense(4, activation = 'linear')(combined_features)
combined_features = BatchNormalization()(combined_features)
combined_features = Activation('relu')(combined_features)
combined_features = Dense(1, activation = 'sigmoid')(combined_features)
similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')
similarity_model.summary()

optimizer = Adam(lr = 0.00006)
similarity_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['mae'])

print(x_train_2.shape[1:])
similarity_model_2 = get_siamese_model(x_train_2.shape[1:])
similarity_model_2.summary()

optimizer = Adam(lr = 0.00006)
similarity_model_2.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['mae'])




def show_model_output(nb_examples = 10):
    pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)
    #pred_sim = similarity_model.predict([pv_a, pv_b])
    pred_sim = similarity_model_2.predict([pv_a, pv_b])
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
    count = 1
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        #ax1.imsave(str(count)+".png", c_a[:,:,0])
        ax1.set_title(' %3.0f%%' % (100*c_d))
        #fig = ax1.get_figure()
        #fig.savefig(str(count)+"A.png")
        print('Image A actual:%3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        
        #fig = ax2.get_figure()
        #fig.savefig(str(count)+".png")
        ax2.set_title('%3.0f%%' % (100*p_d))
        print('Image B predicted:%3.0f%%' % (100*p_d))
        ax2.axis('off')
    return fig
fig = show_model_output()
fig.savefig("siam_first_cosine.png")
def siam_gen(in_groups, batch_size = 10):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim
# we want a constant validation group to have a frame of reference for model performance
valid_a, valid_b, valid_sim = gen_random_batch(test_groups, 10)
print(valid_a.shape)
print("\n")
print(valid_sim.shape)
loss_history = similarity_model_2.fit_generator(siam_gen(train_groups), 
                               steps_per_epoch = 500,
                               validation_data=([valid_a, valid_b], valid_sim),
                                              epochs = 80,
                                             verbose = True)


fig = show_model_output()
fig.savefig("siam_end.png")
