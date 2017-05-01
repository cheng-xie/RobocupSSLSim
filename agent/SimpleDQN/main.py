from environment import Environment
from agent import DQNAgent
import tensorflow as tf
from multiprocessing import Process
from Tkinter import *
import sys, getopt


def main(argv):
    inputfile = None
    train = False

    try:
        opts, args = getopt.getopt(argv,"hrl:",["loadckpt="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -l <ckptfile>'
            sys.exit()
        elif opt == '-r':
            train = True 
        elif opt in ("-l", "--loadckpt"):
            inputfile = arg

    with tf.Session() as sess:
        env = Environment()
        agent = DQNAgent(env, sess, inputfile) 
        if train:
            agent.train(6000000) 
        else:
            agent.test(2000)


if __name__ == '__main__':
    main(sys.argv[1:])
    #p = Process(target=agent.train, args=(600000,)) 
    #p.start() 
