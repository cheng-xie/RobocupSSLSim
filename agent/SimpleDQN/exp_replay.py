import numpy as np
import random

'''
Experience Replay stores pre and post action states for sampling
during training.
'''

class ExperienceReplay:
    'implemented like a circular buffer'
    def __init__(self, state_size):
        # use a current write index implementation for the circular buffer
        self.state_size = state_size
 
        self.current_index = 0
        self.length = 500 
        self.actions = np.empty(self.length, dtype = np.uint8) 
        self.states = np.empty((self.length, self.state_size), dtype = np.float16)
        self.rewards = np.empty(self.length, dtype = np.float16)
        self.dones = np.empty(self.length, dtype = np.bool)
        
    def batch_sample(self, batch_size):
        idxs = [] 
        while len(idxs) < batch_size:
            while True:
                # keep trying random indices
                idx = random.randint(1, self.length - 1) 
                # don't want to grab current index since it wraps 
                if idx == self.current_index and idx == self.current_index - 1:
                    continue 
                idxs.append(idx)
                break
        print('ids', idxs) 
        s_t = self.states[[x-1 for x in idxs]]
        s_t1 = self.states[idxs]
        a_t = self.actions[idxs]
        r_t = self.rewards[idxs]
        done = self.dones[idxs]
        
        return s_t, a_t, r_t, s_t1, done

    def put(self, s_t, a_t, reward, done):
        self.actions[self.current_index] = a_t 
        self.states[self.current_index] = s_t 
        self.rewards[self.current_index] = reward 
        self.dones[self.current_index] = done
        self.current_index = (self.current_index + 1) % self.length 
