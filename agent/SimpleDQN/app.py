from PyQt4 import QtGui, QtCore
import time, sys

class MainAgentWindow(QtGui.QWidget):
    def __init__(self, agent, parent = None):
        super(MainAgentWindow, self).__init__(parent)
        self.button_start = QtGui.QPushButton('Stop', self)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.button_start)
        self.setLayout(self.layout)

        self._agent = agent
        self.thread = AgentThread(self._agent)
        self.thread.run()
        #self.thread.start() 

class AgentThread(QtCore.QThread):
    def __init__(self, agent):
        super(AgentThread, self).__init__()
        self._agent = agent
        self.running = False

    def __del__(self):
        self.stop()
        self.wait()

    def run(self):
        self.running = True
        train_routine = self._agent.train(600000)
        while self.running:
            next(train_routine)

    def stop(self):
        self.running = False

    def update_ep(self, ep):
        self._agent.ep = ep
