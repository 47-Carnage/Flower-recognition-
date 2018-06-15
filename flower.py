import cv2                 
import numpy as np         
import os     
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'             
from random import shuffle 
from tqdm import tqdm      

TRAIN_DIR = 'E:/Minor Project II/FL/train'
TEST_DIR = 'E:/Minor Project II/FL/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'flowers-{}-{}.model'.format(LR, '2conv-basic') 

def label_img(img):
		word_label = img.split('.')[-3]
		if word_label == 'daisy': return [1,0,0,0,0]
		elif word_label == 'dandelion': return [0,1,0,0,0]
		elif word_label == 'rose': return [0,0,1,0,0]
		elif word_label == 'sunflower': return [0,0,0,1,0]
		elif word_label == 'tulip': return [0,0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()

#train_data= np.load('train_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet =max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet,5)

#convnet = conv_2d(convnet, 128, 5, activation='relu')
#convnet = max_pool_2d(convnet,5)

#convnet = conv_2d(convnet, 32, 5, activation='relu')
#convnet = max_pool_2d(convnet,5)

#convnet = conv_2d(convnet, 64, 5, activation='relu')
#convnet = max_pool_2d(convnet,5)

#convnet = conv_2d(convnet, 128, 5, activation='relu')
#convnet = max_pool_2d(convnet,5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)


convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-50]
test = train_data[-50:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=10, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)

import matplotlib.pyplot as plt

test_data = process_test_data()

#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    print(img_num)
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
    model_out = model.predict([data])[0]
    arra = [np.argmax(model_out)]
    if np.argmax(model_out) == 0: str_label='daisy'
    elif np.argmax(model_out) == 1: str_label='dandelion'
    elif np.argmax(model_out) == 2: str_label='rose'
    elif np.argmax(model_out) == 3: str_label='sunflower'
    else: str_label='tulip'
    
    y.imshow(orig,cmap='autumn')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

'''for i in arra:
	label = ['daisy','dandelion','rose','sunflower','tulip']
	str_label = label[i]
	score = data[0][i]
	print('%s (%.5f)' % (str_label,score))'''
