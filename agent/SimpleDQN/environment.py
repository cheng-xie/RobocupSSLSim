
'''
Wrapper around gym environment to process action or state.
'''
class Environment():
    def __init__(self):
        env = gym.make('SSLNav-v0')

    def next_state(self, cur_action):
        self.cur_state, self.cur_reward, self.done, _ = self.env.step(cur_action)
        return self.cur_state, self.cur_reward, self.done

    def new_game(self):
        self.cur_state = self.env.reset()
        return self.cur_state
