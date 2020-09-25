import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import time
import timeit

# import 3 data representation class
from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
from live3DGraph import Live3DGraph
from motion_plan_state import Motion_plan_state

from path_planning.astar_fixLenSOG import astar
# from path_planning.rrt_dubins import RRT
from path_planning.cost import Cost
from path_planning.catalina import create_cartesian

# keep all the constants in the constants.py file
# to get access to a constant, eg:
#   const.SIM_TIME_INTERVAL
import constants as const
import path_planning.catalina as catalina

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


class astarSim:

    def __init__(self, init_x, init_y, init_z, pathLenLimit, weights):
        # initialize auv's data
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.weights = weights

        self.pathLenLimit = pathLenLimit 
        self.live_graph = Live3DGraph()

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


    def display_astar_trajectory(self):
        """
        Display the 2d auv trajectory constructed with A* algorithm

        Parameter:
            NONE 
        """

        start = (self.x, self.y)

        environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS) # output: ([obstacle_list, boundary_list, boat_list, habitat_list])
        
        obstacle_list = environ[0]
        boundary_list = environ[1]
        boat_list = environ[2]
        habitat_list = environ[3]

        # shark_dict = {5: [Motion_plan_state(-120 + (0.3 * i), -50 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)]}
        # astar_solver = astar(start, obstacle_list+boat_list, boundary_list, habitat_list, {}, shark_dict, AUV_velocity=1)
        # final_path_mps = astar_solver.astar(250, weights, shark_dict[5])

        # astar_solver = astar(start, obstacle_list+boat_list, boundary_list, habitat_list) 
        astar_solver = astar(start, obstacle_list+boat_list, boundary_list, habitat_list, {}, AUV_velocity=1)

        # final_path_mps = astar_solver.astar(habitat_list, obstacle_list+boat_list, boundary_list, start, self.pathLenLimit, self.weights)
        final_path_mps = astar_solver.astar(250, self.weights)

        # A_star_traj = final_path_mps[0]
        A_star_traj = final_path_mps["path"]
        # print("\n", "trajectory cost: ", final_path_mps[1][0])
        print ("\n", "Trajectory Cost: ", final_path_mps["cost list"])
        print("\n", "Trajectory: ", A_star_traj)
        # print ("\n", "Trajectory Length: ", len(final_path_mps[0]))
        print ("\n", "Trajectory Length: ", final_path_mps["path length"])
        # A_star_traj_cost = round(final_path_mps[1][0], 2)
        A_star_traj_cost = final_path_mps["cost"]
        # A_star_new_traj = self.create_trajectory_list(A_star_traj)

        astar_x_array = []
        astar_y_array = []

        for point in A_star_traj:
            astar_x_array.append(round(point.x, 2))
            astar_y_array.append(round(point.y, 2))
            # astar_x_array.append(point.x)
            # astar_y_array.append(point.y)

        # print ("\n", "astar_x_array: ", astar_x_array)
        # print ("\n", "astar_y_array: ", astar_y_array)
        self.live_graph.plot_2d_astar_traj(astar_x_array, astar_y_array, A_star_traj_cost)
    
'''
    def display_multiple_astar_trajectory(self):
        """
        Display multiple A* path planned to looking for different goal points

        Parameter:
            NONE 
        """

        start = create_cartesian(catalina.START, catalina.ORIGIN_BOUND)

        x_list = [] # a list holds lists of x coordinates of many trajectories 
        y_list = [] # a list holds lists of y coordinates of many trajectories 

        traj_cost_list = []

        for g in catalina.GOAL_LIST: 

            astar_x_array = []  
            astar_y_array = [] 

            # [obstacle_list, boundary_list, boat_list, habitat_list]
            environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS) 
            goal = catalina.create_cartesian(g, catalina.ORIGIN_BOUND)
            astar_solver = astar(start, goal, environ[0]+environ[2], environ[1]) # should call astar that fixes goal instead of path length
            final_path_mps = astar_solver.astar(environ[3], environ[0]+environ[2], environ[1], start, goal, weights=self.weights)
            traj_cost_list.append(round(final_path_mps[1][0], 2))
            A_star_new_traj = self.create_trajectory_list(final_path_mps[0])

            for point in A_star_new_traj:
                astar_x_array.append(point.x) 
                astar_y_array.append(point.y) 

            x_list.append(astar_x_array)
            y_list.append(astar_y_array)

        self.live_graph.plot_multiple_2d_astar_traj(x_list, y_list, traj_cost_list)
'''

def main():

    pos = create_cartesian((33.446198, -118.486652), catalina.ORIGIN_BOUND)
    test_robot = astarSim(round(pos[0], 2), round(pos[1], 2), 0, pathLenLimit=250, weights=[0, 0, 0, 10])
    test_robot.display_astar_trajectory()

if __name__ == "__main__":
    main()
