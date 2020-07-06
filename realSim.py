import math
import geopy.distance
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon 

import catalina
from motion_plan_state import Motion_plan_state
from live3DGraph import Live3DGraph
from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
import constants as const

from path_planning.rrt_dubins import RRT
from path_planning.astar_real import astar

class RealSim:
    '''
    simulation of real data for path planning
    real data includes information about boundary, obstacles, habitats
    
    integrate with three ways of path planning: RRT, A*, reinforcement learning'''
    
    def __init__(self):
        #initialize start, goal, obstacle, boundary, habitats for path planning
        start = catalina.create_cartesian(catalina.START, catalina.ORIGIN_BOUND)
        self.start = Motion_plan_state(start[0], start[1])

        goal = catalina.create_cartesian(catalina.GOAL, catalina.ORIGIN_BOUND)
        self.goal = Motion_plan_state(goal[0], goal[1])

        self.obstacles = []
        for ob in catalina.OBSTACLES:
            pos = catalina.create_cartesian((ob.x, ob.y), catalina.ORIGIN_BOUND)
            self.obstacles.append(Motion_plan_state(pos[0], pos[1], size=ob.size))
        
        self.boundary = []
        for b in catalina.BOUNDARIES:
            pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
            self.boundary.append(Motion_plan_state(pos[0], pos[1]))
        
        self.boat_list = []
        for boat in catalina.BOATS:
            pos = catalina.create_cartesian((boat.x, boat.y), catalina.ORIGIN_BOUND)
            self.boat_list.append(Motion_plan_state(pos[0], pos[1], size=boat.size))
        
        #testing data for habitats
        self.habitats = []
        for habitat in catalina.HABITATS:
            pos = catalina.create_cartesian((habitat.x, habitat.y), catalina.ORIGIN_BOUND)
            self.habitats.append(Motion_plan_state(pos[0], pos[1], size=habitat.size))
        
        #testing data for shark trajectories
        self.shark_dict = {1: [Motion_plan_state(-102 + (0.1 * i), -91 + (0.1 * i), traj_time_stamp=i) for i in range(1,501)], 
            2: [Motion_plan_state(-150 - (0.1 * i), 0 + (0.1 * i), traj_time_stamp=i) for i in range(1,501)]}
        
        #initialize path planning algorithm
        #A* algorithm
        self.astar = astar(start, goal, self.obstacles+self.boat_list, self.boundary)
        self.A_star_traj = []
        #RRT algorithm
        self.rrt = RRT(self.goal, self.goal, self.boundary, self.obstacles+self.boat_list, self.habitats, freq=10)
        self.rrt_traj = []
        #Reinforcement Learning algorithm

        #initialize live3Dgraph
        self.live3DGraph = Live3DGraph()

        #set up for sharks
        #self.load_real_shark(2)

        self.curr_time = 0

        self.sonar_range = 50
    
    def astar_planning(self):
        '''
        A * path planning algorithm
        call A * planning to generate path from start to goal
        '''
        traj = self.astar.astar(self.obstacles, self.boundary)
        #A * trajectory represented by a list of motion_plan_states
        A_star_new_traj = self.create_trajectory_list_astar(traj)

        #2D x_coordinate list and y_coordinate list for plotting A * trajectory
        astar_x_array = []
        astar_y_array = []

        for point in A_star_new_traj:
            astar_x_array.append(point.x)
            astar_y_array.append(point.y)
        
        self.A_star_traj = [astar_x_array, astar_y_array]   
    
    def rrt_planning(self):
        '''
        RRT path planning algorithm
        call RRT planning to generate optimal path while exploring habitats as much as possible
        '''

        result = self.rrt.exploring(self.shark_dict, 0.5, 5, 1, False, 5.0, 500.0, True, weights=[1,-3,-3,-3])
        #RRT trajectory represented by a list of motion_plan_states
        rrt_traj = result["path"]

        #2D x_coordinate list and y_coordinate list for plotting RRT trajectory
        rrt_x_array = [mps.x for mps in rrt_traj]
        rrt_y_array = [mps.y for mps in rrt_traj]
        self.rrt_traj = [rrt_x_array, rrt_y_array]
    
    def create_trajectory_list_astar(self, traj_list):
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
    
    def get_all_sharks_state(self):
        """
        Return a dictionary representing state for all the sharks 
            key = id of the shark & value = the shark's position (stored as a Motion_plan_state)
        """

        # using dictionary so we can access the state of a shark based on its id quickly?
        self.shark_state_dict = {}
        
        #get shark trajectory as a list of motion_plan_states
        for shark in self.live3DGraph.shark_array:
            #self.live3DGraph.update_shark_location(shark, self.curr_time)
            self.shark_state_dict[shark.id] = shark.get_shark_traj()
    
    def load_real_shark(self, num_shark):
        # load the array of 32 shark trajectories for testing
        shark_testing_trajectories = self.load_shark_testing_trajectories("./data/shark_tracking_data_x.csv", "./data/shark_tracking_data_y.csv")
        shark_id_array = [i+1 for i in range(num_shark)]
        self.live3DGraph.shark_array = list(map(lambda i: shark_testing_trajectories[i],\
            shark_id_array))
        self.live3DGraph.load_shark_labels()
    
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

    def plot_real_traj(self):        
        #trajectory dictionary to store paths for plotting from A *, RRT and RL algorithms
        traj = {"A *" : self.A_star_traj, "RRT" : self.rrt_traj, "RL": []}        
        self.live3DGraph.plot_2d_traj(traj, self.shark_dict)

testing = RealSim()
testing.rrt_planning()
testing.astar_planning()
testing.plot_real_traj()