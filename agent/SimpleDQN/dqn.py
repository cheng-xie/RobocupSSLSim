import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from ops import linear, clipped_error
#from utils import get_time, save_pkl, load_pkl

'''
Model used to predict Q values
'''

class DQN:
    
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.gamma = 0.98
        self.w = {'train': {}, 'target': {}}
        self.dueling = True
        self.state_size = state_size
        self.action_size = action_size
        self.build_QNet()


    def predict_action(self, state):
        # predict the next action
        return self.q_train_action.eval({self.s_t:state})
    
    def get_target_qs(self, state, action_idxs):
        return self.target_q_with_idx.eval({self.t_s_t:state, self.target_q_idx:[[i, act] for i, act in enumerate(action_idxs)]})
    
    def train(self, state, actions, reward, s_t1, done):
        # generate yDQN = r + gamma Q_t(s',argmax Q(s',a'))
        # find argmax Q(s', a')
        predicted_action = self.predict_action(s_t1)
        q_target = self.get_target_qs(s_t1, predicted_action)
        # if we are done just look at the final reward
        yDQN = reward + self.gamma * q_target * (1.-done)
        # feed into optim step
        q_train, _, loss = self.sess.run([self.q_train, self.optim, self.loss],
            {
                self.s_t: state,
                self.action: actions,
                self.yDQN: yDQN,
            }
        )
        return q_train, loss 
    
    def update_target(self):
        # assign each of the weights from train
        for name in self.w['train'].keys():
            self.assign_params_op[name].eval(
                    {self.assign_params[name]: self.w['train'][name].eval()})

    def build_QNet(self):
        self.q_train, self.q_train_action = self._build_train()
        self.q_target, self.target_q_idx, self.target_q_with_idx = self._build_target()
        self._build_update()
        self._build_optim()  
        tf.initialize_all_variables().run()
        self.update_target()
    

    def _build_train(self):
        activation_fn = tf.nn.sigmoid
        with tf.variable_scope('train'):
            # batched s_t to batched q and q_action
            self.s_t = tf.placeholder('float32', [None, self.state_size], name='s_t')

            # MLP Feature Extraction (s_t -> l3)
            l1, self.w['train']['l1_w'], self.w['train']['l1_b'] = linear(self.s_t, 32, activation_fn=activation_fn, name='l1')
            l2, self.w['train']['l2_w'], self.w['train']['l2_b'] = linear(l1, 16, activation_fn=activation_fn, name='l2')
            l3, self.w['train']['l3_w'], self.w['train']['l3_b'] = linear(l2, 16, activation_fn=activation_fn, name='l3')
            if self.dueling:
                # Value Net : V(s) is scalar (l3 -> value)
                value_hid, self.w['train']['l4_val_w'], self.w['train']['l4_val_b'] = linear(l3, 16, activation_fn=activation_fn, name='value_hid')
                value, self.w['train']['val_w_out'], self.w['train']['val_w_b'] = linear(value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action (l3 -> advantage)
                adv_hid, self.w['train']['l4_adv_w'], self.w['train']['l4_adv_b'] = linear(l3, 16, activation_fn=activation_fn, name='adv_hid')
                advantage, self.w['train']['adv_w_out'], self.w['train']['adv_w_b'] = linear(adv_hid, self.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage) Q=V+A-mean(A)
                q_train = value + (advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
            
            else:
                l4, self.w['train']['l4_w'], self.w['train']['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='l4')
                q_train, self.w['train']['q_w'], self.w['train']['q_b'] = linear(l4, self.action_size, name='q')
            
            # Greedy policy
            q_action = tf.argmax(q_train, dimension=1)
            return q_train, q_action

    def _build_target(self):
        activation_fn = tf.nn.sigmoid
        with tf.variable_scope('target'):
            self.t_s_t = tf.placeholder('float32', [None, self.state_size], name='t_s_t')

            # MLP Feature Extraction
            l1, self.w['target']['l1_w'], self.w['target']['l1_b'] = linear(self.t_s_t, 32, activation_fn=activation_fn, name='l1')
            l2, self.w['target']['l2_w'], self.w['target']['l2_b'] = linear(l1, 16, activation_fn=activation_fn, name='l2')
            l3, self.w['target']['l3_w'], self.w['target']['l3_b'] = linear(l2, 16, activation_fn=activation_fn, name='l3')
            if self.dueling:
                # Value Net : V(s) is scalar
                value_hid, self.w['target']['l4_val_w'], self.w['target']['l4_val_b'] = linear(l3, 16, activation_fn=activation_fn, name='value_hid')
                value, self.w['target']['val_w_out'], self.w['target']['val_w_b'] = linear(value_hid, 1, name='value_out')
                
                # Advantage Net : A(s) is vector with advantage given each action
                adv_hid, self.w['target']['l4_adv_w'], self.w['target']['l4_adv_b'] = linear(l3, 16, activation_fn=activation_fn, name='adv_hid')
                advantage, self.w['target']['adv_w_out'], self.w['target']['adv_w_b'] = linear(adv_hid, self.action_size, name='adv_out')

                # Average Dueling (Subtract mean advantage)
                q_target = value + (advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
            
            else:
                l4, self.w['target']['l4_w'], self.w['target']['l4_b'] = linear(l3_flat, 32, activation_fn=activation_fn, name='l4')
                q_target, self.w['target']['q_w'], self.w['target']['q_b'] = linear(l4, self.action_size, name='q')
            
            # The action we use will depend if we use double q learning
            target_q_idx = tf.placeholder('int32', [None, None], name='q_id')
            # Get the q values of the specified state/action indices
            target_q_with_idx = tf.gather_nd(q_target, target_q_idx)
            return q_target, target_q_idx, target_q_with_idx

    def _build_update(self):
        with tf.variable_scope('update_target'):
            self.assign_params = {}
            self.assign_params_op = {}
            for name in self.w['train'].keys():
                self.assign_params[name] = tf.placeholder('float32', self.w['train'][name].get_shape().as_list(), name = name)
                self.assign_params_op[name] = self.w['target'][name].assign(self.assign_params[name])
    
    def _build_optim(self):
        with tf.variable_scope('optimizer'):
            # we can evaluate this seperately since we dont have to propagate errors
            # fed in using r+gammaQ_t(s', argmax Q(s',a'))
            self.yDQN = tf.placeholder('float32', [None], name = 'yDQN')
            
            # find true q for action batch
            self.action = tf.placeholder('int32', [None], name = 'action')
            # batch, features, depth
            action_one_hot = tf.one_hot(self.action, self.action_size, axis = -1)

            # get q values for the action we chose, mask self.q with element wise mult
            # -> q for each batch
            q_for_step = tf.reduce_sum(tf.mul(self.q_train, action_one_hot), 1)

            # get loss from TD
            self.loss = clipped_error(self.yDQN-q_for_step)
            # optimize
            self.optim = tf.train.AdamOptimizer().minimize(self.loss) 

    def clipped_error(x):
        # Huber loss
        try:
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
