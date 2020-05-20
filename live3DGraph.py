import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
Uses matplotlib to generate live 3D Graph while the simulator is running

Able to draw the auv as well as multiple sharks
"""
class Live3DGraph:
    def __init__(self):
        self.shark_array = []
        self.index = 1

        self.colors = ['b', 'g', 'c', 'm', 'y', 'k']

        # initialize the 3d scatter position plot for the auv and shark
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    
    def plot_sharks(self):
        if len(self.shark_array) != 0 and \
            self.index < len(self.shark_array[0].traj_pts_array):
            for i in range(len(self.shark_array)):
                c = self.colors[i % len(self.colors)]
                shark = self.shark_array[i]

                shark.store_positions(shark.traj_pts_array[self.index].x, shark.traj_pts_array[self.index].y, shark.traj_pts_array[self.index].z)
                
                self.ax.plot(shark.x_pos_array, shark.y_pos_array, shark.z_pos_array, marker = 'x', color = c, label = "shark #" + str(shark.id))
                
            self.ax.legend(["auv"] + list(map(lambda s: "shark #" + str(s.id), self.shark_array)))
            self.index += 1
