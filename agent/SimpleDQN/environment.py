import gym
import gym_robossl

'''
Wrapper around gym environment to process action or state.
'''
class Environment():
    def __init__(self):
        self.env = gym.make('SSLNav-v0')
        self.cur_state = None

    def next_state(self, cur_action, render = True):
        self.env.step(cur_action)
        self.env.step(cur_action)
        self.env.step(cur_action)
        self.env.step(cur_action)
        self.cur_state, self.cur_reward, self.done, _ = self.env.step(cur_action)
        if render:
            self.env.render() 
        return self.cur_state, self.cur_reward, self.done

    def new_game(self):
        self.cur_state = self.env.reset()
        self.env.render() 
        return self.cur_state

    @property
    def action_size(self):
        return self.env.action_space.n
