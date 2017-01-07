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
        self.dueling = True
    
    def build_QNet(self):
        with tf.variable_scope('train'):
            # batched s_t to batched q and q_action
            self.s_t = tf.placeholder('float32', [None, self.state_size], name='s_t')

            # MLP Feature Extraction (s_t -> l3)
            self.l1, self.w['train']['l1_w'], self.w['train']['l1_b'] = linear(self.s_t, 256, activation_fn=activation_fn, name='l1')
            self.l2, self.w['train']['l2_w'], self.w['train']['l2_b'] = linear(self.l1, 128, activation_fn=activation_fn, name='l2')
            self.l3, self.w['train']['l3_w'], self.w['train']['l3_b'] = linear(self.l2, 64, activation_fn=activation_fn, name='l3')

            if self.dueling:
                # Value Net : V(s) is scalar (l3 -> value)
                self.value_hid, self.w['train']['l4_val_w'], self.w['train']['l4_val_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='value_hid')
                self.value, self.w['train']['val_w_out'], self.w['train']['val_w_b'] = linear(self.value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action (l3 -> advantage)
                self.adv_hid, self.w['train']['l4_adv_w'], self.w['train']['l4_adv_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='adv_hid')
                self.advantage, self.w['train']['adv_w_out'], self.w['train']['adv_w_b'] = linear(self.adv_hid, self.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage) Q=V+A-mean(A)
                self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            
            else:
                self.l4, self.w['train']['l4_w'], self.w['train']['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
                self.q, self.w['train']['q_w'], self.w['train']['q_b'] = linear(self.l4, self.action_size, name='q')
            
            # Greedy policy
            self.q_action = tf.argmax(self.q, dimension=1)

        with tf.variable_scope('target'):
            self.t_s_t = tf.placeholder('float32', [None, self.state_size], name='s_t')

            # MLP Feature Extraction
            self.l1, self.w['target']['l1_w'], self.w['target']['l1_b'] = linear(self.s_t, 256, activation_fn=activation_fn, name='l1')
            self.l2, self.w['target']['l2_w'], self.w['target']['l2_b'] = linear(self.l1, 128, activation_fn=activation_fn, name='l2')
            self.l3, self.w['target']['l3_w'], self.w['target']['l3_b'] = linear(self.l2, 64, activation_fn=activation_fn, name='l3')

            if self.dueling:
                # Value Net : V(s) is scalar
                self.value_hid, self.w['target']['l4_val_w'], self.w['target']['l4_val_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='value_hid')
                self.value, self.w['target']['val_w_out'], self.w['target']['val_w_b'] = linear(self.value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action
                self.adv_hid, self.w['target']['l4_adv_w'], self.w['target']['l4_adv_b'] = linear(self.l3, 512, activation_fn=activation_fn, name='adv_hid')
                self.advantage, self.w['target']['adv_w_out'], self.w['target']['adv_w_b'] = linear(self.adv_hid, self.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage)
                self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            
            else:
                self.l4, self.w['target']['l4_w'], self.w['target']['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
                self.q, self.w['target']['q_w'], self.w['target']['q_b'] = linear(self.l4, self.action_size, name='q')
            
            # The action we use will depend if we use double q learning
            self.target_q_idx = tf.placeholder('float32', [None, None], name='q_id')
            # Get the q values of the specified state/action indices
            self.target_q_with_idx = tf.gather_nd(self.q, self.target_q_idx)
        
        with tf.variable_scope('update_target'):
            self.assign_params = {}
            self.assign_params_op = {}
            for name in self.w['train'].keys():
                self.assign_params[name] = tf.placeholder('float32', self.w['train'][name].get_shape().as_list(), name = name)
                self.assign_params_op[name] = self.w['target'][name].assign(self.assign_params[name])

        with tf.variable_scope('optimizer'):
            self.yDQN = tf.placeholder('float32', [None], name = 'yDQN')
            
            # find true q from current step
            self.action = tf.placeholder('int64', [None], name = 'action')

            # batch, features, depth
            action_one_hot = tf.one_hot(self.action, self.action_size, axis = -1)
            self.reduce

            # get loss from two

            # optimize


    def clipped_error(x):
        # Huber loss
        try:
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
