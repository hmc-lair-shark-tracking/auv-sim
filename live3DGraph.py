from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Rectangle
from catalina import create_cartesian 
from motion_plan_state import Motion_plan_state

import numpy as np
import constants as const
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import catalina


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
        self.ax = self.fig.add_subplot( 111, projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.arrow_length_ratio = 0.1

        # create a dictionary for checkbox for each type of planned trajectory
        # key - the planner's name: "A *", "RRT"
        # value - three-element array
        #   1. boolean(represent wheter the label is added to the legend)
        #   2. the CheckButtons object
        #   3. color of the plot
        self.traj_checkbox_dict = {}
        # initialize the A * button
        # initialize the A * button
        self.traj_checkbox_dict["A *"] = [False,\
            CheckButtons(plt.axes([0.7, 0.10, 0.15, 0.05]), ["A* Trajectory"]), '#9933ff']
        # when the A* checkbox is checked, it should call self.enable_traj_plot
        
        # initialize the RRT button
        self.traj_checkbox_dict["RRT"] = [False,\
            CheckButtons(plt.axes([0.7, 0.05, 0.15, 0.05]),["RRT Trajectory"]), '#043d10']
        # when the RRT checkbox is checked, it should call self.enable_traj_plot
        self.particle_checkbox = CheckButtons(plt.axes([0.1, 0.10, 0.15, 0.05]),["Display Particles"])
        self.display_particles = False
        self.particle_checkbox.on_clicked(self.particle_checkbox_clicked)

        self.run_sim = True

        self.end_sim_btn = Button(plt.axes([0.1, 0.8, 0.15, 0.05]), "End Simulation")
        self.end_sim_btn.on_clicked(self.end_simulation)

        # an array of the labels that will appear in the legend
        # TODO: labels and legends still have minor bugs
        self.labels = []


    def scale_quiver_arrow(self):
        """
        A hack to work around the bug with quiver plot arrow size

        Manually alter the arrow_length_ratio which affects the arrow size partially
        """
        # The current theory is that the arrow size is weird because the z axis is being autoscaled
        # A work-around is to adjust the arrow size based on the z axis range
        z_interval = self.ax.get_zaxis().get_data_interval()
        range = abs(z_interval[0] - z_interval[1])

        if range == 0:
            # if the distance between the auv and sharks is 0 m in the z direction
            self.arrow_length_ratio = 0.01
        elif range > 50:
            # if the distance between the auv and sharks is greater than 50 m in the z direction
            self.arrow_length_ratio = range * 0.02
        else:
            self.arrow_length_ratio = range * 0.1


    def plot_entity(self, x_pos_array, y_pos_array, z_pos_array, label = 'auv', color = 'red', marker = ','):
        """
        Plot the auv trajectory as well as its direction

        Parameters:
            x_pos_array - an array of floats indicating the auv's past x-position
            y_pos_array - an array of floats indicating the auv's past y-position
            z_pos_array - an array of floats indicating the auv's past z-position
        """

        # calculate the orientation of directino vector
        x_orient = x_pos_array[-1]-x_pos_array[-2]
        y_orient = y_pos_array[-1]-y_pos_array[-2]
        z_orient = z_pos_array[-1]-z_pos_array[-2]

        # plot the trajectory line
        self.ax.plot(x_pos_array, y_pos_array, z_pos_array,\
            marker = marker, linestyle = '-', color = color, label = label)
        
        # use quiver plot to draw an arrow indicating the auv's direction
        self.ax.quiver(x_pos_array[-1], y_pos_array[-1], z_pos_array[-1],\
            x_orient, y_orient, z_orient,\
            color = color, pivot="tip", normalize = True, arrow_length_ratio = self.arrow_length_ratio)


    def load_shark_labels(self):
        """
        Add the sharks that we are tracking to the legend
        
        Should be called in setup() in robotSim after the shark tracking data is loaded
        """
        if len(self.shark_array) != 0:
             # create legend with the auv and all the sharks
            self.labels += list(map(lambda s: "shark #" + str(s.id), self.shark_array))


    def load_auv_labels(self, num_of_auv):
        """
        Add all the auvs to the legend
        
        Should be called in __init__ in robotSim after we create live3DGraph object
        """
        for i in range(1, num_of_auv):
             # create legend with auvs
            self.labels += ["auv #" + str(i)]
    

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

                    self.update_shark_location(shark, sim_time)
                    
                    # calculate orientation by: current coordinate - previous coordinate
                    # these 3 variables will help us indicate the direction of the trajectory
                    x_orient = shark.x_pos_array[-1]-shark.x_pos_array[-2]
                    y_orient = shark.y_pos_array[-1]-shark.y_pos_array[-2]
                    z_orient = shark.z_pos_array[-1]-shark.z_pos_array[-2]

                    # plot the trajectory of the shark
                    self.ax.plot(shark.x_pos_array, shark.y_pos_array, shark.z_pos_array, marker = ",", color = c, label = "shark #" + str(shark.id))

                    # plot the direction vectors for the shark
                    self.ax.quiver3D(shark.x_pos_array[-1], shark.y_pos_array[-1], shark.z_pos_array[-1], x_orient, y_orient, z_orient, color = c, pivot="tip", normalize = True, arrow_length_ratio = self.arrow_length_ratio)
                   

    def update_shark_location(self, shark, sim_time):
        """
        Increment a shark's "index", so we update the position of the shark

        Parameter:

        """
        if shark.index < len(shark.traj_pts_array):
            # increment index variable so we get new position from the shark
            # The shark trajectories have time interval of 0.03s between each trajectory,
            #   but the simulator time interval might be diffent.
            # So we need to increment the index properly so that the newest shark trajectory point is close
            #   to the simulator's current time
            #print("shark traj pts array", shark.traj_pts_array)
            #print("sim_time", sim_time)
            while shark.index < len(shark.traj_pts_array)-1 and\
                abs(shark.traj_pts_array[shark.index].traj_time_stamp - sim_time) > (const.SIM_TIME_INTERVAL + 0.1):
                shark.index += 1
                
            # update the shark's position arrays to help us update the graph
            shark.store_positions(shark.traj_pts_array[shark.index].x, shark.traj_pts_array[shark.index].y, shark.traj_pts_array[shark.index].z)

            
    def enable_traj_plot(self, event):
        """
        Handles when a check box is hit

        event - a string, matches with the name of the label when the checkButton is created
        """
        if (event == "A* Trajectory"):
            # self.traj_checkbox_dict["A *"][0] returns whether the label has been added to 
            #   self.labels aka the legend
            # we only want one copy of the label to be in self.labels
            if not self.traj_checkbox_dict["A *"][0]:
                self.labels += ["A *"]
                self.traj_checkbox_dict["A *"][0] = True
        elif (event == "RRT Trajectory"):
            if not self.traj_checkbox_dict["RRT"][0]:
                self.labels += ["RRT"]
                self.traj_checkbox_dict["RRT"][0] = True        


    def plot_planned_traj(self, planner_name, trajectory_array):
        """
        Plot the planned trajectory specified by the planner name

        Parameters:
            planner_name - string, either "A *" or "RRT"
            trajectory_array - an array of Motion_plan_state objects
        """
        # get the checkbox object
        checkbox = self.traj_checkbox_dict[planner_name][1]
        # boolean, true if the checkbox is checked
        checked = checkbox.get_status()[0]
        # get the color of the trajectory plot (a string representing color in hex)
        color = self.traj_checkbox_dict[planner_name][2]
        
        if checked:
            # self.traj_checkbox_dict["A *"][0] returns whether the label has been added to 
            #   self.labels aka the legend
            # we only want one copy of the label to be in self.labels 
            if not self.traj_checkbox_dict[planner_name][0]:
                self.labels += [planner_name]
                self.traj_checkbox_dict[planner_name][0] = True
            
            '''traj_x_array = []
            traj_y_array = []
            # create two array of x and y positions for plotting
            for traj_pt in trajectory_array:
                traj_x_array.append(traj_pt.x)
                traj_y_array.append(traj_pt.y)'''

            # TODO: for now, we set the z position of the trajectory to be -10
            #self.ax.plot(traj_x_array,  traj_y_array, 0, marker = ',', color = color, label = planner_name)
            self.ax.plot([mps.x for mps in trajectory_array],  [mps.y for mps in trajectory_array], 0, marker = ',', color = color, label = planner_name)
        else:
            # if the checkbox if not checked
            # self.traj_checkbox_dict[planner_name][0] represents whether the label is added to
            #   self.label array
            # we only want to remove the label once
            if self.traj_checkbox_dict[planner_name][0]:
                self.labels.remove(planner_name)
                self.traj_checkbox_dict[planner_name][0] = False
    

    def particle_checkbox_clicked(self, event):
        """
        on_clicked handler function for particle checkbox

        toggle the display_particles variable (bool)
        """
        self.display_particles = not self.display_particles
    
    
    def plot_particles(self, particle_coordinates):
        """
        Plot the particles if the the particle checkbox is checked

        Parameter:
            particle_array - an array of arrays, where each element has the format:
                [x_p, y_p, v_p, theta_p, weight_p]
        """
        if self.display_particles:
            particle_x_array = []
            particle_y_array = []
            particle_color_array = []
            # create two arrays for plotting x and y positions
            for particle in particle_coordinates:
                particle_x_array.append(particle.x_p)
                particle_y_array.append(particle.y_p)
                # particle[4] specify the weight of the points
                # the color of particles based on high weight to low weight:
                #   red -> orange -> purple -> blue
                if particle.weight_p > 0.75 and particle.weight_p <= 1.0:
                    # red
                    particle_color_array.append('#e31263')
                elif particle.weight_p > 0.5 and particle.weight_p <= 0.75:
                    # orange
                    particle_color_array.append('#912951')
                elif particle.weight_p > 0.25 and particle.weight_p <= 0.5:
                    # purple
                    particle_color_array.append('#7a5b67')
                else:
                    # blue
                    particle_color_array.append('#786b70')

            # TODO: for now, we set the z position of the trajectory to be -10

            self.ax.scatter(particle_x_array, particle_y_array, -10, marker = 'o', color = particle_color_array)
            """
            self.ax.set_xlim3d(-75,75)
            self.ax.set_ylim3d(-175,175)
            self.ax.set_zlim3d(-75,75)
            
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            
            """
            self.ax.autoscale(False)

    def shark_plotter_pf(self,new_mean_x, new_mean_y, new_coordinate_x, new_coordinate_y):
        # plots trajectory of shark next to pf
        # plot limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        """
        self.ax.set_xlim3d(-10,10)
        self.ax.set_ylim3d(-10,10)
        self.ax.set_zlim3d(-15,15)
        """
        # my try
        color_list = []
        colors = ['red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue','blueviolet']
        index = -1
        for coordinate in new_coordinate_x:
            index += 1 
            if index == 6:
                index = -1
            color_list.append(colors[index])
            # particles
        self.ax.scatter(new_mean_x, new_mean_y, -15, marker = 'o', color = color_list)
        self.ax.scatter(new_coordinate_x, new_coordinate_y, -15, marker = 'o', color = color_list)
        self.ax.plot(new_mean_x, new_mean_y, -15, marker = 'o', color = '#286330')
        self.ax.plot(new_coordinate_x, new_coordinate_y, -15, marker = 'o', color = '#1b1c1b')
        #mean particles --> green
        # shark position --> blue

    
    def plot_obstacles(self, obstacle_array, color='#000000'):
        """
        Plot obstacles as sphere based on location and size indicated by the "obstacle_array"

        Parameter - obstacle_array
            an array of motion_plan_states that represent the obstacles's
                position and size
        """
        for obs in obstacle_array:
            # number of points used to plot the sphere
            # the higher N is, the more refined will the obstacles look
            #   (but at the expense of taking longer time)
            N = 50
            
            u = np.linspace(0, 2 * np.pi, N)
            v = np.linspace(0, np.pi, N)
            x = obs.size * np.outer(np.cos(u), np.sin(v)) + obs.x
            y = obs.size * np.outer(np.sin(u), np.sin(v)) + obs.y
            z = obs.size * np.outer(np.ones(np.size(u)), np.cos(v)) + obs.z

            self.ax.plot_surface(x, y, z, linewidth=0.0, cstride = 1, rstride = 1, color = color, alpha = 0.2)  

    
    def end_simulation(self, events):
        """
        End the live simulation (terminate the main navigation while loop in robotSim)
        """
        self.run_sim = False


    def plot_distance(self, all_dist_dict, time_array):
        """
        Use to generate one of the summary plots
        
        Parameters:
            all_dist_dict - a dictionary storing the distance between the auv and sharks
                key: shark id & value: an array storing the distance
            time_array - an array for all the time stamps
        """  
        # allow us to have multiple subplots (differnt summary graphs) in the future
        plt.subplots()

        for shark_id in all_dist_dict:
            label = "shark #" + str(shark_id)
            plt.plot(time_array, all_dist_dict[shark_id], label=label)

        plt.xlabel('x - time (sec)')
        plt.ylabel('y - distance between auv and shark (m)')
        plt.title('distance between auv and all the sharks during simulation')

        plt.legend()


    def plot_2d_sim_graph(self, auv_x_array, auv_y_array, obstacle_array=[]):
        """
        Plot the 2d summary graph of the overall trajectory for auv and sharks and also the obstacles

        Parameters:
            auv_x_array - an array of floats, indicating the auv x position throughout the simulation
            auv_y_array - an array of floats, indicating the auv y position throughout the simulation
            obstacle_array - (optional) an array of motion_plan_states that represent the obstacles's
                position and size
        """
        # close the 3D simulation plot (if there's any)
        plt.close()
        
        fig, ax = plt.subplots()
        
        # plot the auv overall trajectory
        
        
        # calculate the orientation of direction vector for the auv
        for i in range(len(auv_x_array)):
            x_orient = auv_x_array[i][-1]-auv_x_array[i][-2]
            y_orient = auv_y_array[i][-1]-auv_y_array[i][-2]
            plt.plot(auv_x_array[i], auv_y_array[i], marker = ',', color = 'r', label='auv')
            # plot an arrow indicating the auv direction
            plt.quiver(auv_x_array[i][-1], auv_y_array[i][-1], x_orient, y_orient, color = 'r', pivot="tail")

        # plot all the sharks
        if len(self.shark_array) != 0:         
            for i in range(len(self.shark_array)):
                
                    # determine the color of this shark's trajectory
                c = self.colors[i % len(self.colors)]

                shark = self.shark_array[i]
                plt.plot(shark.x_pos_array, shark.y_pos_array, marker = ",", color = c, label = "shark #" + str(shark.id))

                # calculate orientation by: current coordinate - previous coordinate
                # these 3 variables will help us indicate the direction of the trajectory
                x_orient = shark.x_pos_array[-1]-shark.x_pos_array[-2]
                y_orient = shark.y_pos_array[-1]-shark.y_pos_array[-2]

                plt.quiver(shark.x_pos_array[-1], shark.y_pos_array[-1], x_orient, y_orient, color = c, pivot="tail")

        # plot all the obstacles
        for obs in obstacle_array:
            ax.add_patch(plt.Circle((obs.x, obs.y), obs.size, color = '#000000', fill = False))

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Summary of the auv and shark trajectories during the simulation')

        plt.legend()

        plt.show()

    def plot_2d_traj(self, traj_dict, shark_dict):
        """
        Plot a trajectory with defined boundaries and obstacles
        """

        plt.close()
        
        fig, ax = plt.subplots()

        # plot the boundaries as polygon lines
        Path = mpath.Path
        path_data = []

        for i in range(len(catalina.BOUNDARIES)): 
            pos = create_cartesian((catalina.BOUNDARIES[i].x, catalina.BOUNDARIES[i].y), catalina.ORIGIN_BOUND)
            if i == 0: 
                path_data.append((Path.MOVETO, pos))
            else:
                path_data.append((Path.LINETO, pos))

        last = create_cartesian((catalina.BOUNDARIES[0].x, catalina.BOUNDARIES[0].y), catalina.ORIGIN_BOUND)
        path_data.append((Path.CLOSEPOLY, last))

        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=None, alpha=0)

        ax.add_patch(patch) 

        # plot obstacels as circles 
        for obs in catalina.OBSTACLES:
            pos_circle = create_cartesian((obs.x, obs.y), catalina.ORIGIN_BOUND)
            ax.add_patch(plt.Circle(pos_circle, obs.size, color = '#000000', fill = False))
        
        # plot boats as circles
        for boat in catalina.BOATS:
            pos_boat = create_cartesian((boat.x, boat.y), catalina.ORIGIN_BOUND)
            ax.add_patch(plt.Circle(pos_boat, boat.size, color = '#000000', fill = False))
        
        for habitat in catalina.HABITATS:
            pos_habitat = create_cartesian((habitat.x, habitat.y), catalina.ORIGIN_BOUND)
            ax.add_patch(plt.Circle(pos_habitat, habitat.size, color = 'b', fill = False))
        
        x, y = zip(*path.vertices)
        line, = ax.plot(x, y, 'go-')

        ax.grid()
        ax.axis('equal')

        # plot trajectory
        for planner, traj in traj_dict.items():
            if traj != []:
                color = self.traj_checkbox_dict[planner][2]
                ax.plot(traj[0], traj[1], marker = ',', color = color, label=planner)

        # plot sharks
        for shark_id, shark_pos in shark_dict.items():
            ax.plot([mps.x for mps in shark_pos],[mps.y for mps in shark_pos], label=shark_id)
        ax.legend()
        plt.show()
