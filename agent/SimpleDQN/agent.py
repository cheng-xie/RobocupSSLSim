import random
import copy
from dqn import DQN
from exp_replay import ExperienceReplay
import numpy as np
from time import sleep

class DQNAgent:
    
    def __init__(self, env, sess, load_path = None):
        self.env = env
        self.sess = sess
        self.ep = 0.35
        self.state_size = 6
        self.batch_size = 128 
        self.start_train = 370100 
        self.train_freq = 10 
        self.action_size = self.env.action_size 
        self.qnet = DQN(self.sess, self.state_size, self.action_size, load_path)
        self.xrep = ExperienceReplay(self.state_size) 
        print 'made agent'

    def store_exp(self, next_state, action, reward, done):
        self.xrep.put(next_state, action, reward, done)

    def train(self, numsteps, save_path):
        # fill experience replay
        # start the process
        print 'game'
        self.env.new_game()
       
        for self.step in xrange(self.start_train):
            # Get env state
            cur_state = self.env.cur_state
            # Act on the environment
            cur_action = self.get_next_action(cur_state, 1)
            
            next_state, reward, done = self.env.next_state(cur_action, render = (self.step % 1500 == 0)) 
            # Save experience
            self.store_exp(next_state, cur_action, reward, done)

            if done:
                # start a new game
                self.env.new_game()
            
            #yield self.step
        
        num_trains = 0
        

        tot_q = 0
        tot_reward = 0
        tot_loss = 0
        for self.step in xrange(numsteps):
            # Get env state
            cur_state = self.env.cur_state
            # Act on the environment
            cur_action = self.get_next_action(cur_state, self.ep)
            
            next_state, reward, done = self.env.next_state(cur_action, render = False)#(self.step % 800 == 0)) 
            # Save experience
            self.store_exp(next_state, cur_action, reward, done)
            
            tot_reward += reward

            if self.step % self.train_freq == 0:
                #print(cur_action, cur_state)
                # Train Q-network 
                q, loss = self.train_network() 
                num_trains += 1
                tot_q += np.mean(q)
                tot_loss += np.sum(loss)
                if num_trains % 1800 == 0:
                    self.store_exp(next_state, cur_action, reward, True)
                    # Update target network
                    print("step:{:8d} ep:{:05.4f} q:{:10.4f} l:{:10.4f} r:{:10.4f}".format(self.step, self.ep, tot_q, tot_loss, tot_reward/18./self.train_freq))
                    self.ep = max(self.ep*0.97-0.0003,0)
                    self.update_target()
                    tot_q = 0
                    tot_loss = 0
                    tot_reward = 0
                    self.save_weights(save_path)
                    for _ in xrange(100):
                        cur_state = self.env.cur_state
                        cur_action = self.get_next_action(cur_state, eps = 0.01)
                        _,_,done1 = self.env.next_state(cur_action, render = False) 
                        if done1:
                            # start a new game
                            self.env.new_game()

                        cur_state = self.env.cur_state
                        cur_action = self.get_next_action(cur_state, eps = 0.01)
                        _,_,done1 = self.env.next_state(cur_action, render = True) 
                        if done1:
                            # start a new game
                            self.env.new_game()

            if done:
                # start a new game
                self.env.new_game()
            
            #yield self.step

    def test(self, num_steps):
        self.env.new_game()
        for self.step in xrange(num_steps):
            cur_state = self.env.cur_state
            cur_action = self.get_next_action(cur_state, eps = 0.00)
            _,_,done1 = self.env.next_state(cur_action, render = True) 
            sleep(0.08)
            if done1:
                # start a new game
                self.env.new_game()


    def update_target(self):
        self.qnet.update_target()

    def train_network(self):
        # sample from exp_replay
        s_t, action, reward, s_t1, done = self.xrep.batch_sample(self.batch_size)
        #print(s_t,action,s_t1,done)
        #print(s_t.shape,action.shape,s_t1.shape,done.shape)
        #print('Reward', reward, np.mean(reward))
        q, loss = self.qnet.train(s_t, action, reward, s_t1, done)  
        return q, loss

    def get_next_action(self, cur_state, eps=1):
        if random.random() < eps:
            action = random.randrange(self.env.action_size)
        else:
            action = self.qnet.predict_action(cur_state.reshape(1,-1))[0]
        return action

    def save_weights(self, path):
        self.qnet.save_net(path)

    def load_weights(self, path):
        self.qnet.load_net(path)
