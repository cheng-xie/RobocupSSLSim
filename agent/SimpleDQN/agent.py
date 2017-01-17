from dqn import DQN
from exp_replay import ExperienceReplay


class DQNAgent:
    
    def __init__():
        self.discounting_factor = 0.9
        self.qnet = DQN()
        self.xrep = ExperienceReplay() 


    def store_exp(self, cur_state, new_state, reward, action):
        exp_replay.put(cur_state, new_state, reward, action)

    def train():
        for self.step in xrange(start, end):
            # Get env state
            cur_state =
            # Act on the environment
            cur_action = self.get_next_action(cur_state) 
            next_state = self.env.act(cur_action) 
            # Save experience
            self.store_exp(cur_state, next_state, reward, cur_action)
            # Train Q-network 
            q, loss = self.train_network() 
            # Update target network
            self.update_target()

    def test():
        for self.step in xrange(start, end):
            # Get env state
            cur_state =
            # Act on the environment
            cur_action = self.get_next_action(cur_state) 

    def update_target(self):
        self.qnet.update_target()

    def train_network(self):
        # sample from exp_replay
        s_t, action, reward, s_t1, done = self.memory.sample()
        q, loss = self.qnet.train(s_t, action, reward, s_t1, done)  
        return q, loss

    def get_next_action(self, cur_state):
        action = self.qnet.predict_action(cur_state)
        return action       

    def get_target_qs(self, ):
        q_target = self.qnet.
        return 
