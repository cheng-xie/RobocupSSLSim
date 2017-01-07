import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .replay_memory import ReplayMemory
from .ops import linear, conv2d, clipped_error
from utils import get_time, save_pkl, load_pkl

'''
Model used to predict Q values
'''

class DQN:
    
    def __init__(self):
        self.w = {'train': {}, 'target': {}}
         
    
    def build_QNet(self, context = 'train'):
        with tf.variable_scope(context):
            self.s_t = tf.placeholder('float32', [None, self.state_size], name='s_t')

            # MLP Feature Extraction
            self.l1, self.w['l1_w'], self.w['l1_b'] = linear(self.s_t, 256, activation_fn=activation_fn, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = linear(self.l1, 128, activation_fn=activation_fn, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.l2, 64, activation_fn=activation_fn, name='l3')

            if self.dueling:
                # Value Net : V(s) is scalar
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='value_hid')
                self.value, self.w['val_w_out'], self.w['val_w_b'] = linear(self.value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action
                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='adv_hid')
                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = linear(self.adv_hid, self.env.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage)
                self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            
            else:
                self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
                self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
            
            # Greedy policy
            self.q_action = tf.argmax(self.q, dimension=1)
        
        with tf.variable_scope('target'):
            self.s_t = tf.placeholder('float32', [None, self.state_size], name='s_t')

            # MLP Feature Extraction
            self.l1, self.w['l1_w'], self.w['l1_b'] = linear(self.s_t, 256, activation_fn=activation_fn, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = linear(self.l1, 128, activation_fn=activation_fn, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.l2, 64, activation_fn=activation_fn, name='l3')

            if self.dueling:
                # Value Net : V(s) is scalar
                self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='value_hid')
                self.value, self.w['val_w_out'], self.w['val_w_b'] = linear(self.value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action
                self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='adv_hid')
                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = linear(self.adv_hid, self.env.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage)
                self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            
            else:
                self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
                self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
            
            # Greedy policy
            self.q_action = tf.argmax(self.q, dimension=1)

    def batch_train(self, s0, a, r, s1, d):
        
