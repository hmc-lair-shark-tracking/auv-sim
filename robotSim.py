import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

import gym

import time
import timeit
import random
import numpy

from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
from live3DGraph import Live3DGraph
from motion_plan_state import Motion_plan_state
from auv import Auv
from particleFilter import ParticleFilter

#import path planning class
from path_planning.astar import astar
from path_planning.sharkOccupancyGrid import SharkOccupancyGrid
from path_planning.rrt_dubins import RRT
from path_planning.cost import Cost
#from catalina import create_cartesian

# keep all the constants in the constants.py file
# to get access to a constant, eg:
#   const.SIM_TIME_INTERVAL
import constants as const
import catalina 


def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians
    """
    if -math.pi <= ang <= math.pi:
        return ang
    elif ang > math.pi: 
        ang += (-2 * math.pi)
        return angle_wrap(ang)
    elif ang < -math.pi: 
        ang += (2 * math.pi)
        return angle_wrap(ang)


class RobotSim:
    def __init__(self, init_x, init_y, init_z, init_theta, replan_time, planning_time, num_of_auv, curr_time):
        # initialize auv's data
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.theta = init_theta
        self.num_of_auv = num_of_auv

        # need lists to keep track of all the points if we want to
        #   connect lines between the position points and create trajectories
        self.x_list = [init_x]
        self.y_list = [init_y]
        self.z_list = [init_z]

        self.shark_sensor_data_dict = {}

        # keep track of the current time that we are in
        # each iteration in the while loop will be assumed as 0.1 sec
        self.curr_time = curr_time
        self.curr_time = 0
        self.time_array = []
        
        # keep track when there will be new sensor data of sharks
        # start out as 20, so particle filter will get some data in the beginning
        self.sensor_time = const.NUM_ITER_FOR_NEW_SENSOR_DATA

        # index for which trajectory point that we should
        # keep track of
        self.curr_traj_pt_index = 0

        # create a square trajectory (list of motion_plan_state object)
        # with parameter: v = 1.0 m/s and delta_t = 0.5 sec
        #self.testing_trajectory = self.get_auv_trajectory(5, 0.5)

        self.live_graph = Live3DGraph()
        # create proper labels for the auvs, so the legend will work
        self.live_graph.load_auv_labels(num_of_auv)

        #time interval to replan trajectory
        self.replan_time = 0.5
        #time limit for path planning algorithm to find the shortest path
        self.planning_time = planning_time


    def get_auv_state(self):
        """
        Return a Motion_plan_state representing the orientation and the time stamp
        of the robot
        """
        return Motion_plan_state(self.x, self.y, theta = self.theta, traj_time_stamp=self.curr_time)


    def get_all_sharks_state(self):
        """
        Return a dictionary representing state for all the sharks 
            key = id of the shark & value = the shark's position (stored as a Motion_plan_state)
        """

        # using dictionary so we can access the state of a shark based on its id quickly?
        shark_state_dict = {}

        #print(self.live_graph.shark_array)
        for shark in self.live_graph.shark_array:
            shark_state_dict[shark.id] = shark.get_curr_position()
        
        return shark_state_dict

    """
    def get_auv_sensor_measurements(self):
        
        Return an Motion_plan_state object that represents the measurements
            of the auv's x,y,z,theta position with a time stamp
        The measurement has random gaussian noise
        
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution

        # np.random.normal returns a single sample drawn from the parameterized normal distribution
        # we actually omitted the third parameter which determines the number of samples that we would like to draw

        return Motion_plan_state(x = self.x + np.random.normal(0,1),\
            y = self.y + np.random.normal(0,1),\
            z = self.z + np.random.normal(0,1),\
            theta = angle_wrap(self.theta + np.random.normal(0,1)),\
            traj_time_stamp = self.curr_time)

    """

    def get_all_sharks_sensor_measurements(self, shark_state_dict, auv_sensor_data):
        """
        Modify the data member self.shark_state_dict if there is new sensor data
            key = id of the shark & value = the shark's range and bearing (stored as a sharkState object)

        Parameter: 
            shark_state_dict - a dictionary, containing the shark's states at a given time
            auv_sensor_data - a motion_plan_state object, containting the auv's position

        Return Value:
            True - if it has been 2 sec and there are new shark sensor measurements
            False - if it hasn't been 2 sec
        """
        # decide to sensor_time an integer because floating point addition is not as reliable
        # each iteration through the main navigation loop is 0.1 sec, so 
        #   we need 20 iterations to return a new set of sensor data
        if self.sensor_time == const.NUM_ITER_FOR_NEW_SENSOR_DATA:
            # iterate through all the sharks that we are tracking
            for shark_id in shark_state_dict: 
                shark_data = shark_state_dict[shark_id]
                print("shark_data.x", shark_data.x)
                delta_x = shark_data.x - auv_sensor_data.x
                delta_y = shark_data.y - auv_sensor_data.y
                print(" x ",delta_x)
                range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
                bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

                Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
                Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) + bearing_random)

                self.shark_sensor_data_dict[shark_id] = SharkState(shark_data.x, shark_data.y, Z_shark_range, shark_id)
            
            # reset the 2 sec time counter
            self.sensor_time = 0
            
            return True
        else: 
            self.sensor_time += 1

            return False

    def get_all_sharks_sensor_measurements(self, shark_state_dict, auv_sensor_data):
        """
        Modify the data member self.shark_state_dict if there is new sensor data
            key = id of the shark & value = the shark's range and bearing (stored as a sharkState object)

        Parameter: 
            shark_state_dict - a dictionary, containing the shark's states at a given time
            auv_sensor_data - a motion_plan_state object, containting the auv's position

        Return Value:
            x,y of the shark
        """
        # decide to sensor_time an integer because floating point addition is not as reliable
        # each iteration through the main navigation loop is 0.1 sec, so 
        #   we need 20 iterations to return a new set of sensor data
        if self.sensor_time == const.NUM_ITER_FOR_NEW_SENSOR_DATA:
            # iterate through all the sharks that we are tracking
            for shark_id in shark_state_dict: 
                shark_data = shark_state_dict[shark_id]

                delta_x = shark_data.x - auv_sensor_data.x
                delta_y = shark_data.y - auv_sensor_data.y
                
                range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
                bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

                Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
                Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) + bearing_random)

                self.shark_sensor_data_dict[shark_id] = SharkState(shark_data.x, shark_data.y, Z_shark_range, Z_shark_bearing, shark_id)
            
            # reset the 2 sec time counter
            self.sensor_time = 0
            
            return True
        else: 
            self.sensor_time += 1

            return False
        
        
    def get_habitats(self):
        '''
        get the location of all habitats within the boundary, represented as a list of motion_plan_states
        '''

        habitats = []

        #testing habitat lists
        habitats = [Motion_plan_state(750, 300, size=5), Motion_plan_state(750, 320, size=2), Motion_plan_state(780, 240, size=10),\
            Motion_plan_state(775, 330, size=3), Motion_plan_state(760, 320, size=5), Motion_plan_state(770, 250, size=4),\
            Motion_plan_state(800, 295, size=4), Motion_plan_state(810, 320, size=5), Motion_plan_state(815, 300, size=5),\
            Motion_plan_state(825, 330, size=6), Motion_plan_state(830, 335, size=5)]
        
        return habitats


    def track_trajectory(self, trajectory, new_trajectory):
        """
        Return an Motion_plan_state object representing the trajectory point TRAJ_LOOK_AHEAD_TIME sec ahead
        of current time

        Parameters: 
            trajectory - a list of trajectory points, where each element is 
            a Motion_plan_state object that consist of time stamp, x, y, z,theta
        """
        # only increment the index if it hasn't reached the end of the trajectory list
        if new_trajectory:
            self.curr_traj_pt_index = 0

        while (self.curr_traj_pt_index < len(trajectory)-1) and\
            (self.curr_time + const.TRAJ_LOOK_AHEAD_TIME) > trajectory[self.curr_traj_pt_index].traj_time_stamp: 
            self.curr_traj_pt_index += 1

        return trajectory[self.curr_traj_pt_index]


    def calculate_new_auv_state(self, v, w, delta_t):
        """ 
        Calculate new x, y and theta

        Parameters: 
            v - linear velocity of the robot (m/s)
            w - angular veloctiy of the robot (rad/s)
            delta_t - time step (sec)
        """
        self.x = self.x + v * math.cos(self.theta)*delta_t
        self.y = self.y + v * math.sin(self.theta)*delta_t
        self.theta = angle_wrap(self.theta + w * delta_t)

        self.x_list += [self.x]
        self.y_list += [self.y]
        self.z_list += [self.z]


    def send_trajectory_to_actuators(self, v, w):
        # TODO: For now this should just update AUV States?
        # 0.5 = const.SIM_TIME_INTERVAL
        self.calculate_new_auv_state(v, w, 0.5)

    def replan_trajectory(self, planner, auv_pos, shark_pos, obstacle, boundary, habitats):
        '''after replan_time, calculate a new trajectory based on planner chosen
        
        Parameters:
            planner: path planning algorithm: RRT or A*
            auv_pos: current auv position
            shark_pos: current shark position
            obstacle: obstacle list
            boundary'''
        
        if planner == "RRT":
            path_planning = RRT(auv_pos, shark_pos, boundary, obstacle, habitats)

            result = path_planning.exploring(habitats, 0.5, 5, 1)
            
            return result["path"]
    
    def create_trajectory_list(self, traj_list):
        """
        Run this function to update the trajectory list by including intermediate positions 

        Parameters:
            traj_list - a list of Motion_plan_state, represent positions of each node
        """
        time_stamp = 0.1
        
        #constant_gain = 1
    
        trajectory_list = []

        step = 0 

        for i in range(len(traj_list)-1):
            
            trajectory_list.append(traj_list[i])

            x1 = traj_list[i].x
            y1 = traj_list[i].y
            x2 = traj_list[i+1].x 
            y2 = traj_list[i+1].y

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            dist = math.sqrt(dx**2 + dy**2)

            velocity = 1

            time = dist/velocity
            
            counter = 0 

            while counter < time:
                if x1 < x2:
                    x1 += time_stamp * velocity
                if y1 < y2:
                    y1 += time_stamp * velocity
                
                step += time_stamp
                counter += time_stamp
                trajectory_list.append(Motion_plan_state(x1, y1, traj_time_stamp=step))
                
            trajectory_list.append(traj_list[i+1])
            
        return trajectory_list

    def log_data(self):
        """
        Print in the terminal (and possibly write the 
        data in a log file?)
        """

        print("AUV [x, y, z, theta]:  [", self.x, ", " , self.y, ", ", self.z, ", ", self.theta, "]")


    def plot(self,  x_list, y_list, z_list, show_live_graph = True, planned_traj_array = [], particle_array = [], obstacle_array = []):
        """
        Wrapper function for plotting and updating shark position

        If we are not showing live graph, it will call the update_shark_location function to 
            update the shark's location without plotting

        Parameters:
            show_live_graph - boolean, determine wheter the live 3d graph should be drawn or not
             planned_traj_array - (optional) an array of trajectories that we want to plot
                each element is an array on its own, where
                    1st element: the planner's name (either "A *" or "RRT")
                    2nd element: the list of Motion_plan_state returned by the planner
            particle_array - (optional) an array of particles
                each element has this format:
                    [x_p, y_p, v_p, theta_p, weight_p]
            obstacle_array - (optional) an array of motion_plan_states that represent the obstacles's
                position and size
        """
        index = -1
        
        if show_live_graph:
            self.update_live_graph(x_list, y_list, z_list, planned_traj_array, particle_array[index], obstacle_array[index])
        else:
            for shark in self.live_graph.shark_array:
                # only update the shark's position without plotting them
                self.live_graph.update_shark_location(shark, self.curr_time)


    def update_live_graph(self, x_list, y_list, z_list, planned_traj_array = [], particle_array = [], obstacle_array = []):
        """
        Plot the position of the robot, the sharks, and any planned trajectories

        Parameter: 
            planned_traj_array - (optional) an array of trajectories that we want to plot
                each element is an array on its own, where
                    1st element: the planner's name (either "A *" or "RRT")
                    2nd element: the list of Motion_plan_state returned by the planner
            particle_array - (optional) an array of particles
                each element has this format:
                    [x_p, y_p, v_p, theta_p, weight_p]
            obstacle_array - (optional) an array of motion_plan_states that represent the obstacles's
                position and size
        """
        # scale the arrow for the auv and the sharks properly for graph
        self.live_graph.scale_quiver_arrow()

        # plot the auv
        self.live_graph.plot_entity(self.x_list, self.y_list, self.z_list)


        # plot the new positions for all the sharks that the robot is tracking
        self.live_graph.plot_sharks(self.curr_time)
        
        # if there's any planned trajectory to plot, plot each one
        if planned_traj_array != []:
            for auv_planned_trajs in planned_traj_array:
                for traj in auv_planned_trajs:
                    # pass in the planner name and the trajectory array
                    self.live_graph.plot_planned_traj(traj[0], traj[1])

        # if there's particles to plot, plot them
        
        if particle_array != []:
            self.live_graph.plot_particles(particle_array)
        
        if obstacle_array != []:
            self.live_graph.plot_obstacles(obstacle_array)

        self.live_graph.ax.legend(self.live_graph.labels)

        # self.live_graph.plot_obstacles(self.get_habitats(), color="red")
        
        # re-add the labels because they will get erased
        self.live_graph.ax.set_xlabel('X')
        self.live_graph.ax.set_ylabel('Y')
        self.live_graph.ax.set_zlabel('Z')

        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.5)

        self.live_graph.ax.clear()


    def summary_graphs(self):
        """
        Generate summary plot(s) after the "End Simulation" button is clicked
        """
        # diction where each value stores an array represanting the distance between the auv and a shark
        auv_all_sharks_dist_dict = {}

        for shark in self.live_graph.shark_array:
            # store a list of distances between the auv and the shark at each time-stamp
            dist_array = []
            for i in range(len(self.x_list)-2):
                delta_x = shark.x_pos_array[i] - self.x_list[i]
                delta_y = shark.y_pos_array[i] - self.y_list[i]

                dist_array.append(math.sqrt(delta_x**2 + delta_y**2))
            
            auv_all_sharks_dist_dict[shark.id] = dist_array
        
        # close the 3D simulation plot (if there's any)
        plt.close()

        # plot the distance between auv and sharks over time graph
        self.live_graph.plot_distance(auv_all_sharks_dist_dict, self.time_array)

        plt.show()


    def render_for_rl_env(self, auv_pos, shark_pos):
        self.auv_x_array_rl.append(auv_pos[0])
        self.auv_y_array_rl.append(auv_pos[1])
        self.auv_z_array_rl.append(auv_pos[2])

        self.shark_x_array_rl.append(shark_pos[0])
        self.shark_y_array_rl.append(shark_pos[1])
        self.shark_z_array_rl.append(shark_pos[2])

        self.live_graph.plot_entity(self.auv_x_array_rl, self.auv_y_array_rl, self.auv_z_array_rl, label = 'auv', color = 'r', marker = ',')

        self.live_graph.plot_entity(self.shark_x_array_rl, self.shark_y_array_rl, self.shark_z_array_rl, label = 'shark', color = 'b', marker = 'o')

        self.live_graph.ax.legend()
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.5)

        self.live_graph.ax.clear()


    def track_way_point(self, way_point):
        """
        Calculates the v&w to get to the next point along the trajectory

        way_point - a motion_plan_state object, represent the trajectory point that we are tracking
        """
        # K_P and v are stand in values
        K_P = 0.5
        # v = 12
        v = 1  # TODO: currently change it to a very unrealistic value to show the final plot faster
       
        angle_to_traj_point = math.atan2(way_point.y - self.y, way_point.x - self.x) 
        w = K_P * angle_wrap(angle_to_traj_point - self.theta) #proportional control
        
        return v, w
    

    def get_auv_trajectory(self, v, delta_t):
        """
        Create an array of trajectory points representing a square path

        Parameters:
            v - linear velocity of the robot (m/s)
            delta_t - the time interval between each time stamp (sec)
        """
        traj_list = []
        t = 0
        x = 760
        y = 300
        z = -10

        for i in range(20):
            x = x + v * delta_t
            y = y
            theta = 0
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,traj_time_stamp=t))

        for i in range(20):
            x = x
            y = y + v * delta_t
            theta = math.pi/2
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,traj_time_stamp=t))
    
        for i in range(20):
            x = x - v * delta_t
            y = y 
            theta = math.pi
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,traj_time_stamp=t))

        for i in range(20):
            x = x
            y = y - v * delta_t 
            theta = -(math.pi)/2
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,traj_time_stamp=t))

        return traj_list


    def load_shark_testing_trajectories(self, x_pos_filepath, y_pos_filepath):
        """
        Load shark tracking data from the csv file specified by the filepath
        Store all the trajectories in an array of SharkTrajectory objects
            SharkTrajectory contains an array of trajectory points with x and y position of the shark
        
        Parameter:
            x_pos_filepath - a string, represent the path to the x position csv data file
            y_pos_filepath - a string, represent the path to the y position csv data file
        """
        shark_testing_trajectories = []

        all_sharks_x_pos_array = []
        all_sharks_y_pos_array = []

        # store the x position for all the sharks
        with open(x_pos_filepath, newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',') 

            for row in data_reader:
                all_sharks_x_pos_array.append(row)
        
        # store the y position for all the sharks
        with open(y_pos_filepath, newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',') 

            for row in data_reader:
                all_sharks_y_pos_array.append(row)

        # create shark trajectories for all the sharks
        for shark_id in range(len(all_sharks_x_pos_array)):
            shark_testing_trajectories.append(SharkTrajectory(shark_id, all_sharks_x_pos_array[shark_id], all_sharks_y_pos_array[shark_id]))
        
        return shark_testing_trajectories


    def setup(self, x_pos_filepath, y_pos_filepath, shark_id_array = []):
        """
        Run this function if we want to track sharks based on their trajectory data in csv file

        Parameters:
            x_pos_filepath - a string, represent the path to the x position csv data file
            y_pos_filepath - a string, represent the path to the y position csv data file
            shark_id_array - an array indicating the id of sharks we want to track
                eg. for the sharkTrackingData.csv (with 32 sharks), the available ids have the range [0, 31]
        """
        # load the array of 32 shark trajectories for testing
        shark_testing_trajectories = self.load_shark_testing_trajectories(x_pos_filepath, y_pos_filepath)
        
        # based on the id of the shark, build an array of shark that we will track 
        # for this simulation
        self.live_graph.shark_array = list(map(lambda i: shark_testing_trajectories[i],\
            shark_id_array))
        
        self.live_graph.load_shark_labels()
    

    def check_terminate_cond(self):
        """
        Check if the main navigation loop should terminate automatically
        2 possible terminating condition
            - if the auv and one of the shark is with in the TERMINATE_DISTANCE range
            - if the simulator reaches the MAX_TIME

        Return:
            True - if the main navigation loop should terminate
        """
        reach_any_shark = False
        reach_max_time = False


        if self.curr_time > const.MAX_TIME:
            reach_max_time = True
            return reach_max_time

        for shark in self.live_graph.shark_array:
            delta_x = shark.x_pos_array[-1] - self.x_list[-1]
            delta_y = shark.y_pos_array[-1] - self.y_list[-1]

            distance = math.sqrt(delta_x**2 + delta_y**2)
            
            if distance <= const.TERMINATE_DISTANCE:
                reach_any_shark = True
                break
        
        return reach_any_shark or reach_max_time


    def main_navigation_loop(self, show_live_graph = True):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data

        Parameter:
            show_live_graph - boolean (option), True if the simulator should show the 3D graph
        """
        #set start time
        t_start = self.curr_time
        # somewhere here add initialize PF 
        for filter in sorted(self.filter_dict):
            # particleFilter object created
            test_particle = self.filter_dict[filter]
            # dictionary for range and bearings
            measurement_data_dict = {}
            #created particles
            particles = test_particle.create()
            while self.live_graph.run_sim:
                print("current time")
                print(self.curr_time)
                final_planned_traj_array = []
                final_auv_x_array = []
                final_auv_y_array = []
                final_auv_z_array = []
                final_obstacle_array = []
                final_particle_array = []
                # update particles
                particles = test_particle.create_and_update(particles)
                shark_state_dict = self.get_all_sharks_state()

                all_auvs_range_bearing_dict = []
                measurement_dict_list = []
                auv_index = -1
                for auv in sorted(self.auv_dict):
                    auv_index += 1
                    test_auv = self.auv_dict[auv]
                    auv_sensor_data = self.auv_dict[auv].get_auv_sensor_measurements(self.curr_time)
                    measurement_dict_list.append(test_auv.get_all_sharks_sensor_measurements(shark_state_dict, auv_sensor_data))
                    # updates the particleFitlers's auv x and y 
                final_measurement_dict_list = []
                # this makes sure that the particleFitler is only getting range and bearing information from the first shark
                for measurement in measurement_dict_list:
                    final_measurement_dict_list.append(measurement[0])
                # update particleFilter shark's x and y positions, will change this after, need shark coordinates to update in order to calculate range error 

                for measurement in final_measurement_dict_list:
                    test_particle.x_shark = measurement[0]
                    test_particle.y_shark = measurement[1]
                    
                particles = test_particle.update_weights(particles,final_measurement_dict_list)
                
                xy_mean = test_particle.particleMean(particles)
                range_error = test_particle.meanError(xy_mean[0], xy_mean[1])

                for auv in sorted(self.auv_dict):
                    test_auv = self.auv_dict[auv]
                    
                    # have to store the information from all the AUVS Range and bearings 
                    
                    # in "measurement_dict[2].range", I have to add the measurement_dict[1].range of the other AUVs information... 

                    # need suggestions on how to store this information...
                    """
                    #boolean  (new data)and dictionary (data) of True and False and Updates Shark Dictionary--> stores range and bearing of the auv
                    if has_new_data == True:
                        print("======NEW DATA=======")
                        print("All The Shark Sensor Measurements [range, bearing]: " +\
                            str(self.shark_sensor_data_dict))
                        for shark in sorted(self.shark_sensor_data_dict):
                            test_shark = self.shark_sensor_data_dict[shark]
                            measurement_dict[shark] = [test_shark.range, test_shark.bearing]
                    else:
                        for shark in sorted(self.shark_sensor_data_dict):
                            test_shark = self.shark_sensor_data_dict[shark]
                            measurement_dict[shark] = [test_shark.range, test_shark.bearing]
                        #print("measurement dict", measurement_dict)
                    """
                    # example of how to indicate the obstacles and plot them

                    obstacle_array = [Motion_plan_state(757,243, size=2),Motion_plan_state(763,226, size=5)]
                    # testing data for plotting RRT_traj
                    boundary = [Motion_plan_state(-500, -500), Motion_plan_state(500,500)]
                    #testing data for habitats
                    habitats = [Motion_plan_state(63,23, size=5), Motion_plan_state(12,45,size=7), Motion_plan_state(51,36,size=5), Motion_plan_state(45,82,size=5),\
                        Motion_plan_state(60,65,size=10), Motion_plan_state(80,79,size=5),Motion_plan_state(85,25,size=6)]
                    #condition to replan trajectory
                    """
                    if self.curr_time == 0 or self.curr_time - t_start >= self.replan_time:
                        RRT_traj = self.replan_trajectory("RRT", auv_sensor_data, shark_state_dict[1], obstacle_array, boundary, habitats)
                        new_trajectory = True
                        t_start = self.curr_time
                    else:
                        new_trajectory = False
                    """
                    RRT_traj = [Motion_plan_state(0.0, 0.0)]
                    RRT_traj += [Motion_plan_state(i, i) for i in range(50)]
                    new_trajectory = False
                    # test trackTrajectory
                    tracking_pt = self.auv_dict[auv].track_trajectory(RRT_traj, new_trajectory , self.curr_time)
                    """
                    print("==================")
                    print ("Currently tracking point: " + str(tracking_pt))
                    """
                    #v & w to the next point along the trajectory
                    (v, w) = self.auv_dict[auv].track_way_point(tracking_pt)
                    '''print("==================")
                    print ("v and w: ", v, ", ", w)
                    print("====================================")
                    print("====================================")'''
                    test_auv.send_trajectory_to_actuators()
                    # update the auv position    
                    # self.log_data()

                    # testing data for plotting A_star_traj
                    A_star_traj = [Motion_plan_state(0.0, 0.0)]
                    A_star_traj += [Motion_plan_state(i, i) for i in range(50)]

                    # example of first parameter to update_live_graph function
                    #planned_traj_array = [["A *", A_star_traj], ["RRT", RRT_traj]]
                    planned_traj_array = []

                    # testing data for displaying particle array
                    
                    particle_array = particles

                    # example of first parameter to update_live_graph function
                    planned_traj_array = [["A *", A_star_traj], ["RRT", RRT_traj]]

                    # In order to plot your planned trajectory, you have to wrap your trajectory in another array, where
                    #   1st element: the planner's name (either "A *" or "RRT")
                    #   2nd element: the list of Motion_plan_state returned by your planner
                    # Use the "planned_traj_array" as an example
                    
                    obstacle_array = []
                    final_planned_traj_array.append(planned_traj_array)
                    final_auv_x_array.append(test_auv.x_list)
                    final_auv_y_array.append(test_auv.y_list)
                    final_auv_z_array.append(test_auv.z_list)
                    final_particle_array.append(particle_array)
                    final_obstacle_array.append(obstacle_array)

                self.time_array.append(self.curr_time)
                # increment the current time by 0.1 second
                self.curr_time += const.SIM_TIME_INTERVAL

                self.plot(final_auv_x_array, final_auv_y_array, final_auv_z_array, show_live_graph, final_planned_traj_array, final_particle_array, final_obstacle_array)
                
                terminate_loop = self.check_terminate_cond()
                if terminate_loop:
                    self.live_graph.run_sim = False
                    break
                    
            obstacle_array = [Motion_plan_state(757,243, size=10), Motion_plan_state(763,226, size=15)]
            self.live_graph.plot_2d_sim_graph(final_auv_x_array, final_auv_y_array, obstacle_array)
            
            # "End Simulation" button is pressed, generate summary graphs for this simulation
            # self.summary_plots()


def main():
    #pos = create_cartesian(catalina.START, catalina.ORIGIN_BOUND)
    test_robot = RobotSim(5.0, 5.0, 0, 0.1, 0.5, 1, 3, 0)
    BOUNDARY_RANGE = 500
    """
    for i in range(test_robot.num_of_auv - 1):
        x = random.uniform(- BOUNDARY_RANGE, BOUNDARY_RANGE)
        y = random.uniform(- BOUNDARY_RANGE, BOUNDARY_RANGE)
        z = random.uniform(- BOUNDARY_RANGE, BOUNDARY_RANGE)
        theta = 0
        velocity = random.uniform(0,4)
        curr_traj_pt_index = 0
        w_1 = random.uniform(-math.pi/2, math.pi/2)
    
        test_robot.auv_dict[i + 1] = Auv(x,y,z, theta, velocity, w_1, curr_traj_pt_index, i)
    """
    theta = 0
    velocity = 1/2
    w_1 = random.uniform(-math.pi/2, math.pi/2)
    curr_traj_pt_index = 0
    test_robot.auv_dict[1] = Auv(100,0,10, theta, velocity, w_1, curr_traj_pt_index, 1)
    test_robot.auv_dict[2] = Auv(0,150,10, theta, velocity, w_1, curr_traj_pt_index, 2)
    # load shark trajectories from csv file
    # the second parameter specify the ids of sharks that we want to track
    test_robot.setup("./data/shark_tracking_data_x.csv", "./data/shark_tracking_data_y.csv", [1,2])
    shark_state_dict = test_robot.get_all_sharks_state()
    # create a dictionary of all the particleFilters
    shark_state = shark_state_dict[1]
    auv_state = [test_robot.auv_dict[1], test_robot.auv_dict[2]]
    #list of auv objects 
    test_robot.filter_dict[0] = ParticleFilter(shark_state.x, shark_state.y, auv_state)
    
    test_robot.main_navigation_loop()

    # test_robot.display_auv_trajectory()
    
    # to not show that live_graph, you can pass in "False"
    test_robot.main_navigation_loop(True)


if __name__ == "__main__":
    main()