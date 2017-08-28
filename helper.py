import numpy as np
import random
import scipy.misc
import os
import csv
import itertools
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont


def ensure_dir(file_path):
    '''
    Used to ensure to create the a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
#
# #Used to initialize weights for policy and value output layers
# def normalized_columns_initializer(std=1.0):
#     def _initializer(shape, dtype=None, partition_info=None):
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#     return _initializer


class RingBuffer():
    """
    TO-TEST: test the sampling procedure
    """
    def __init__(self, length):
        self.buffer = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.buffer.size
        self.buffer[x_index] = x
        self.index = x_index[-1] + 1

    def sample(self, size):
        return np.random.choice(self.buffer, size)


class ExperienceBuffer():
    def __init__(self, buffer_size=50000):
        '''
        store a history of experiences that can be randomly drawn from when training the network. We can draw form the
        previous past experiment to learn
        :param buffer_size: size of the buffer
        '''
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience)) + len(list(self.buffer))) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def set_image_gridworld(frame,measurements,step,goal,hero):
    b = np.ones([840,640,3]) * 255.0
    b = Image.fromarray(b.astype('uint8'))
    draw = ImageDraw.Draw(b)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((240, 670),'Step: ' + str(step),(0,0,0),font=font)
    draw.text((240, 720),'Battery: ' + str(measurements[1]),(0,0,0),font=font)
    draw.text((240, 770),'Deliveries: ' + str(measurements[0]),(0,0,0),font=font)
    c = np.array(b)
    drone = np.array(Image.open('./resources/drone.png'))
    c[hero[0]*128:hero[0]*128+128,hero[1]*128:hero[1]*128+128,:] = drone
    battery = np.array(Image.open('./resources/battery.png'))
    c[0:128,0:128,:] = battery
    house = np.array(Image.open('./resources/house.png'))
    c[goal[0]*128:goal[0]*128+128,goal[1]*128:goal[1]*128+128,:] = house
    return c

def set_image_gridworld_reward(frame,reward,step,goal,hero):
    b = np.ones([840,640,3]) * 255.0
    b = Image.fromarray(b.astype('uint8'))
    draw = ImageDraw.Draw(b)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((240, 670),'Step: ' + str(step),(0,0,0),font=font)
    draw.text((240, 720),'Deliveries: ' + str(reward),(0,0,0),font=font)
    c = np.array(b)
    drone = np.array(Image.open('./resources/drone.png'))
    c[hero[0]*128:hero[0]*128+128,hero[1]*128:hero[1]*128+128,:] = drone
    house = np.array(Image.open('./resources/house.png'))
    c[goal[0]*128:goal[0]*128+128,goal[1]*128:goal[1]*128+128,:] = house
    return c
