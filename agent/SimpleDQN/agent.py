import random
from dqn import DQN
from exp_replay import ExperienceReplay
import numpy as np

class DQNAgent:
    
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.ep = 0.20
        self.state_size = 8
        self.batch_size = 128
        self.start_train = 700 
        self.train_freq = 500 
        self.action_size = self.env.action_size 
        self.qnet = DQN(self.sess, self.state_size, self.action_size)
        self.xrep = ExperienceReplay(self.state_size) 

    def store_exp(self, cur_state, action, reward, done):
        self.xrep.put(cur_state, action, reward, done)

    def train(self, numsteps):
        # fill experience replay
        # start the process
        self.env.new_game()
        num_trains = 0
        for self.step in xrange(numsteps):
            # Get env state
            cur_state = self.env.cur_state
            # Act on the environment
            cur_action = self.get_next_action(cur_state)
            next_state, reward, done = self.env.next_state(cur_action) 
            # Save experience
            self.store_exp(cur_state, cur_action, reward, done)
            if self.step > self.start_train:
                if self.step % self.train_freq == 0:
                    print(cur_action, cur_state)
                    # Train Q-network 
                    q, loss = self.train_network() 
                    num_trains += 1 
                
                if num_trains % 100 == 99:
                    # Update target network
                    self.update_target()

            if done:
                # start a new game
                self.env.new_game()
    '''
    def test():
        for self.step in xrange(start, end):
            # Get env state
            cur_state = self.env.
            # Act on the environment
            cur_action = self.get_next_action(cur_state) 
    '''

    def update_target(self):
        self.qnet.update_target()

    def train_network(self):
        # sample from exp_replay
        s_t, action, reward, s_t1, done = self.xrep.batch_sample(self.batch_size)
        print(s_t,action,s_t1,done)
        #print(s_t.shape,action.shape,s_t1.shape,done.shape)
        print('Reward', reward, np.mean(reward))
        q, loss = self.qnet.train(s_t, action, reward, s_t1, done)  
        return q, loss

    def get_next_action(self, cur_state):
        if random.random() < self.ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.qnet.predict_action(cur_state.reshape(1,-1))[0]
        
        return action
