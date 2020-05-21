# add all the stuff we gotta import
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform 

class particleFilter:
    # 2 sets of initial data- shark's initial position and velocity, and position of AUV 
    # output- estimates the sharks position and velocity
    def __init__(self, init_x, init_y, init_theta):
        self.x = init_x
        self.y = init_y
        self.theta = init_theta
        self.__setat klandflknalksnddf
