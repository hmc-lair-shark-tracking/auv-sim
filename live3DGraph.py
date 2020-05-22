import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons

"""
Uses matplotlib to generate live 3D Graph while the simulator is running

Able to draw the auv as well as multiple sharks
"""
class Live3DGraph:
    def __init__(self):
        self.shark_array = []

        # array of pre-defined colors, 
        # so we can draw sharks with different colors
        self.colors = ['b', 'g', 'c', 'm', 'y', 'k']

        # initialize the 3d scatter position plot for the auv and shark
        self.fig = plt.figure(figsize = [13, 10])
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.added_A_star_label = False
        self.added_RRT_label = False
        
        self.traj_checkbox_dict = {}
        
        self.traj_checkbox_dict["A *"] = [False,\
            CheckButtons(plt.axes([0.7, 0.10, 0.1, 0.05]), ["A* Trajectory"]), '#9933ff']
        self.traj_checkbox_dict["A *"][1].on_clicked(self.enable_traj_plot)
        
        self.traj_checkbox_dict["RRT"] = [False,\
            CheckButtons(plt.axes([0.7, 0.05, 0.1, 0.05]),["RRT Trajectory"]), '#043d10']
        self.traj_checkbox_dict["RRT"][1].on_clicked(self.enable_traj_plot)

        self.labels = ["auv"]


    def load_shark_labels(self):
        if len(self.shark_array) != 0:
             # create legend with the auv and all the sharks
            self.labels += list(map(lambda s: "shark #" + str(s.id), self.shark_array))
    

    def plot_sharks(self, sim_time):
        """
        Plot the trajectory of all the sharks that the robot is 
        tracking in this simulation
        """
        # check if there is any shark to draw
        # and if we have already looped through all the trajectory points
        if len(self.shark_array) != 0:         
            for i in range(len(self.shark_array)):
                if self.shark_array[0].index < len(self.shark_array[0].traj_pts_array):
                    # determine the color of this shark's trajectory
                    c = self.colors[i % len(self.colors)]
                    shark = self.shark_array[i]
                    
                    while shark.index < len(shark.traj_pts_array) and\
                        abs(shark.traj_pts_array[shark.index].time_stamp - sim_time) > 0.2:
                        shark.index += 1

                    # update the shark's position arrays to help us update the graph
                    shark.store_positions(shark.traj_pts_array[shark.index].x, shark.traj_pts_array[shark.index].y, shark.traj_pts_array[shark.index].z)
                    
                    self.ax.plot(shark.x_pos_array, shark.y_pos_array, shark.z_pos_array, marker = ',', color = c, label = "shark #" + str(shark.id))

            
    def enable_traj_plot(self, event):
        if (event == "A* Trajectory"):
            if not self.traj_checkbox_dict["A *"][0]:
                self.labels += ["A *"]
                self.traj_checkbox_dict["A *"][0] = True
        elif (event == "RRT Trajectory"):
            if not self.traj_checkbox_dict["RRT"][0]:
                self.labels += ["RRT"]
                self.traj_checkbox_dict["RRT"][0] = True        


    def plot_planned_traj(self, planner_name, trajectory_array):
        checkbox = self.traj_checkbox_dict[planner_name][1]
        checked = checkbox.get_status()[0]
        color = self.traj_checkbox_dict[planner_name][2]
        
        if checked: 
            traj_x_array = []
            traj_y_array = []
            for traj_pt in trajectory_array:
                traj_x_array.append(traj_pt.x)
                traj_y_array.append(traj_pt.y)

            self.ax.plot(traj_x_array,  traj_y_array, -10, marker = ',', color = color, label = planner_name)
        else:
            if self.traj_checkbox_dict[planner_name][0]:
                self.labels.remove(planner_name)
                self.traj_checkbox_dict[planner_name][0] = False
