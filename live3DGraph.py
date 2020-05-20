import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Uses matplotlib to generate live 3D Graph while the simulator is running

Able to draw the auv as well as multiple sharks
"""
class live3DGraph:
    def __init__(self):
        self.shark_list = []
        self.shark_animation_counter = 0
        
        # initialize the 3d scatter position plot for the auv and shark
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def plot_shark(self):
