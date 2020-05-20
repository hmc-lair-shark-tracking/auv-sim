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
        
        # initialize the 3d scatter position plot for the auv and shark
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.shark_plot_array = []

        self.colors = ['b', 'g', 'c', 'm', 'y', 'k']

    def init_shark_plots(self):
        for shark in self.shark_array:
            self.shark_plot_array.append(
                self.ax.plot([shark.trajectory_array[0].x], [shark.trajectory_array[0].y], [shark.trajectory_array[0].z], marker = "x", color="blue"))    
        print("finishes initialization")
        print(self.shark_plot_array)
    
    # def plot_sharks(self):
    #     print("plot sharks")

    #     if self.index < len(self.shark_array[0].trajectory_array):
    #         for i in range(len(self.shark_array)):
    #             c = self.colors[i % len(self.colors)]
    #             shark = self.shark_array[i]
    #             # print(type(shark))
    #             # print(shark)
    #             # print(type(shark.trajectory_array[self.index].x))
    #             # print(shark.trajectory_array[self.index].x)
    #             self.ax.plot([shark.trajectory_array[self.index].x], [shark.trajectory_array[self.index].y], [shark.trajectory_array[self.index].z], marker = 'x', color = c)
        
    #         self.index += 1
    def plot_sharks(self):
        print("plot sharks")

        for i in range(len(self.shark_plot_array)):
            curr_plot = self.shark_plot_array[i][0]
            curr_shark = self.shark_array[i]
            print(type(curr_plot))
            print(curr_plot)
            x_data, y_data, z_data = curr_plot.get_data_3d()
            print(x_data)
            print(curr_shark.trajectory_array[self.index].x)
            x_data = np.append(x_data, [curr_shark.trajectory_array[self.index].x])
            print(x_data)
            y_data = np.append(y_data, [curr_shark.trajectory_array[self.index].y])
            z_data = np.append(z_data, [curr_shark.trajectory_array[self.index].z])

            curr_plot.set_data_3d(x_data, y_data, z_data)
            # curr_plot.set_xdata(curr_plot.get_xdata().append(curr_shark.trajectory_array[self.counter].x))
            # curr_plot.set_ydata(curr_plot.get_ydata().append(curr_shark.trajectory_array[self.counter].y))
            print("added plot")
        
        self.index += 1

