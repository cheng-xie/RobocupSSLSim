from dqn import DQN
from exp_replay import ExperienceReplay


class DQNAgent:
    
    def __init__(self, env):
        self.qnet = DQN()
        self.xrep = ExperienceReplay() 
        self.env = env

    def store_exp(self, cur_state, new_state, reward, action):
        xrep.put(cur_state, new_state, reward, action)

    def train(numsteps):
        # fill experience replay
        # start the process
        for self.step in xrange(numsteps):
            # Get env state
            cur_state = self.env.cur_state
            # Act on the environment
            cur_action = self.get_next_action(cur_state) 
            next_state, reward, done = self.env.act(cur_action) 
            # Save experience
            self.store_exp(cur_state, reward, cur_action)
            # Train Q-network 
            q, loss = self.train_network() 
            # Update target network
            self.update_target()

            if done:
                # start a new game
                self.env.new_game()

    def test():
        for self.step in xrange(start, end):
            # Get env state
            cur_state = self.env.
            # Act on the environment
            cur_action = self.get_next_action(cur_state) 

    def update_target(self):
        self.qnet.update_target()

    def train_network(self):
        # sample from exp_replay
        s_t, action, reward, s_t1, done = self.xrep.sample()
        q, loss = self.qnet.train(s_t, action, reward, s_t1, done)  
        return q, loss

    def get_next_action(self, cur_state):
        action = self.qnet.predict_action(cur_state)
        return action
