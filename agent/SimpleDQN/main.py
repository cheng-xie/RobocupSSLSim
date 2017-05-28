from environment import Environment
from agent import DQNAgent
import tensorflow as tf
from multiprocessing import Process
from Tkinter import *
import sys, getopt


def main(argv):
    # Pretrained network to use
    inputfile = None
    # Wether to train or to test
    train = False
    # Trained network
    outputfile = None
    
    try:
        opts, args = getopt.getopt(argv,"hrl:s:",["loadckpt=","saveckpt="])
    except getopt.GetoptError:
        print 'Incorrect usage. For more information: test.py -h'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python test.py -r -l <ckptfile> -s <ckptfile>'
            print '-r for enabling training'
            print '-l for loading pre-existing model'
            print '-s for saving  model to file'
            sys.exit()
        elif opt == '-r':
            train = True 
        elif opt in ("-l", "--loadckpt"):
            inputfile = arg
        elif opt in ("-s", "--saveckpt"):
            outputfile = arg

    with tf.Session() as sess:
        env = Environment()
        agent = DQNAgent(env, sess, inputfile) 
        if train:
            agent.train(6000000, outputfile) 
        else:
            agent.test(2000)


if __name__ == '__main__':
    main(sys.argv[1:])
    #p = Process(target=agent.train, args=(600000,)) 
    #p.start() 
