


class DQNAgent:
    
    def __init__():
        self.qnet = DQN() 

    def get_next_action(self, cur_state):
        action = self.qnet.predict(cur_state)
        return action

    def store_exp(self, cur_state, new_state, reward, action):
        exp_replay.put(cur_state, new_state, reward, action)

    def train():
        for self.step in xrange(start, end):
            # Get env state
            cur_state =
            # Act on the environment
            cur_action = self.get_next_action() 
            next_state = self.env.act(cur_action) 
            # Save experience
            self.store_exp(cur_state, next_state, reward, cur_action)
            # Train Q-network 
                      
            # Update target network
   
    def train_network():


    def test():
        
