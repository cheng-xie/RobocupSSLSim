from environment import Environment
from agent import DQNAgent
import app
import tensorflow as tf
from multiprocessing import Process
from Tkinter import *
from PyQt4 import QtGui, QtCore

if __name__ == '__main__':
    with tf.Session() as sess:
        env = Environment()
        agent = DQNAgent(env, sess)
        mainApp = QtGui.QApplication(sys.argv) 
        mainw = app.MainAgentWindow(agent)
        mainw.show()
        sys.exit(mainApp.exec_())
        #p = Process(target=agent.train, args=(600000,)) 
        #p.start() 
        #agent.train(6000000) 
