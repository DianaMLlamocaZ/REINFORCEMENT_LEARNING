from collections import deque
import random

class replay_buffer():
    def __init__(self,length):
        self.__buffer=deque(iterable=[],maxlen=length)

    def push(self,transition):  #Agrego la transición de cada paso
        self.__buffer.append(transition)

    def sample(self,batch_dimen):
        return random.sample(self.__buffer,batch_dimen)

    def length(self):  #Ver el size del buffer
        return len(self.__buffer)
    
    def show_last(self): #Muestro la última transición
        return self.__buffer[-1]
