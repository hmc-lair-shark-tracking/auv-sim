import random
import timeit
import csv 
import matplotlib.pyplot as plt
import numpy as np
import catalina

from astar_fixLenSOG import singleAUV
from motion_plan_state import Motion_plan_state
from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon 
from sharkOccupancyGrid import SharkOccupancyGrid, splitCell
from matplotlib import cm, patches, collections
from astar_fixLenSOG import createSharkGrid
from cost import Cost

class multiAUV:
    def __init__(self, start, numAUV, habitatList, boundaryList, obstacleList, sharkGrid):
        start_position = catalina.create_cartesian(start, catalina.ORIGIN_BOUND)
        self.start = (round(start_position[0], 2), round(start_position[1], 2))
        self.habitat_open_list = habitatList.copy()
        self.habitat_closed_list = []
        self.boundary_list = boundaryList
        self.obstacle_list = obstacleList
        self.numAUV = numAUV

        coords = []
        for corner in boundaryList: 
            coords.append((corner.x, corner.y))
        # initialize shark occupancy grid
        self.boundary_poly = Polygon(coords)
        # divide the workspace into cells
        self.cell_list = splitCell(self.boundary_poly, 10)  
        if sharkGrid == {}:
            self.sharkGrid = createSharkGrid('path_planning/shark_data/AUVGrid_prob_500_straight.csv', self.cell_list)
        else:
            self.sharkGrid = sharkGrid

    def multi_AUV(self, pathLenLimit, weights):
        """
        Find the optimal path for each AUV  
        Parameter: 
            pathLenLimit: in meters; the length limit of the A* trajectory 
            weights: a list of three numbers [w1, w2, w3] 
            shark_traj_list: a list of shark trajectories that are lists of Motion_plan_state objects 
        """

        for i in range(self.numAUV):
            single_AUV = singleAUV(self.start, self.obstacle_list, self.boundary_list, self.habitat_list, self.sharkGrid, AUV_velocity=1)
            single_planner = single_AUV.astar(pathLenLimit, weights)

            print ("\n", "path ", i+1, ": ", single_planner["path"])
            print ("\n", "path length ", i+1, ": ", single_planner["path length"])
            print ("\n", "path cost ", i+1, ": ", single_planner["cost"])
            print ("\n", "path cost list ", i+1, ": ", single_planner["cost list"])

if __name__ == "__main__":
    start = (33.446198, -118.486652)
    numAUV = 3
    pathLenLimit = 100 
    weights = [0, 10, 10, 100]
    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    obstacle_list = environ[0] + environ[2]
    boundary_list = environ[1]
    habitat_list = environ[3]
    sharkGrid = {}
    shark_dict = {1: [Motion_plan_state(-120 + (0.3 * i), -60 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
        2: [Motion_plan_state(-65 - (0.3 * i), -50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
        3: [Motion_plan_state(-110 + (0.3 * i), -40 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
        4: [Motion_plan_state(-105 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
        5: [Motion_plan_state(-120 + (0.3 * i), -50 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
        6: [Motion_plan_state(-85 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
        7: [Motion_plan_state(-270 + (0.3 * i), 50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
        8: [Motion_plan_state(-250 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
        9: [Motion_plan_state(-260 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
        10: [Motion_plan_state(-275 + (0.3 * i), 80 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)]}  
    boundary_poly = []
    for pos in boundary_list:
        boundary_poly.append((pos.x, pos.y))

    boundary = Polygon(boundary_poly) # a Polygon object that represents the boundary of our workspace 
    sharkOccupancyGrid = SharkOccupancyGrid(shark_dict, 10, boundary, 50, 50)
    grid_dict = sharkOccupancyGrid.convert()

    AUVs = multiAUV(start, numAUV, habitat_list, boundary_list, obstacle_list, sharkGrid)
    AUVs.multi_AUV(pathLenLimit, weights)