from environment import Environment
from agent import DQNAgent
import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        env = Environment()
        agent = DQNAgent(env, sess) 
        agent.train(3000000) 
