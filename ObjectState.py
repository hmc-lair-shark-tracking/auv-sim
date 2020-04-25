import math
#import matplotlib.pyplot as plt
import time  # temporarily import time so the code can sleep
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

noise = np.random.normal(0,1,100)

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise
class ObjectState:
    def __init__(self, x, y, theta, time_stamp):
        # initialize auv's data
        self.x = x
        self.y = y
        self.theta = theta
        self.time_stamp = time_stamp 

    def get_auv_sensor_measurements(self):
        Z_auv = ObjectState(self.x, self.y, self.theta)
        Z_auv.x = self.x + noise
        Z_auv.y = self.y + noise
        Z_auv.theta = self.theta + noise
        return Z_auv

