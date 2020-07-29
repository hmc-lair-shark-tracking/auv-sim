import math
import time
import statistics
import random
import decimal
import catalina
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost 
from astar_fixLenSOG import astar, Node 
from shapely.geometry import Polygon
from rrt_dubins import RRT, createSharkGrid
from motion_plan_state import Motion_plan_state
from statistics import mean 

def cal_boundary_peri(astar_solver):
    peri = 0
    for i in range(len(astar_solver.boundary_list)-1):
        dist, _ = astar_solver.get_distance_angle(astar_solver.boundary_list[i], astar_solver.boundary_list[i+1])
        peri += dist

    return peri

def get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, habitats, sharkGrid, weights):
    """
    Return the mean cost of the produced path

    Parameter: 
        trajectory: a list of Node objects
    """
    total_cost = 0
    num_steps = 0
    max_time = traj_node[-1].time_stamp 
    time = 0 
    cal_cost = Cost()
    while time < max_time:
        currPath_mps = get_curr_mps_path(time, traj_mps) 
        currPath_node = get_curr_node_path(time, traj_node)
        cost = cal_cost.habitat_shark_cost_func(currPath_mps, currPath_node[-1].pathLen, peri_boundary, total_traj_time, habitats, sharkGrid, weights)
        # print ("habitat_shark_cost = ", cost[0], "currPath length = ", len(currPath_mps), "currPath_node[-1].pathLen = ", currPath_node[-1].pathLen)
        total_cost += cost[0]
        time += 2 # dt = 2s
        num_steps += 1 
    mean_cost = total_cost/num_steps
    print ("mean cost = ", total_cost, " / ", num_steps, " = ", mean_cost)

    return mean_cost

def get_curr_mps_path(curr_time_stamp, traj_mps):
    """
    Return the path that A* generates up until the curr_time_stamp

    Parameter:
        curr_time_stamp: an integer
        traj_mps: a list of Motion_plan_state objects
    """

    current_path = []
    for mps in traj_mps:
        if mps.traj_time_stamp <= curr_time_stamp:
            # print ("mps.traj_time_stamp = ", mps.traj_time_stamp, "curr_time_stamp = ", curr_time_stamp)
            current_path.append(mps)
            # print ("current_path = ", current_path)
        else:
            break

    return current_path

def get_curr_node_path(curr_time_stamp, traj_node):
    """
    Return the path that A* generates up until the curr_time_stamp

    Parameter:
        curr_time_stamp: an integer
        traj_mps: a list of Node objects
    """
    
    current_path = []

    for node in traj_node: 
        if node.time_stamp <= curr_time_stamp:
            current_path.append(node)
        else:
            break

    return current_path

def build_environ(complexity): 
    """
    Generate the right number of randomized obstacles and habitats corresponding to the complexity level

    Parameter:
        complexity: an integer; represents the number of obstacles and habitats
    """

    habitats = []
    obstacles = []

    test_a = []
    test_b = []

    for i in range(complexity):
        obs_x = float(decimal.Decimal(random.randrange(33443758, 33445914))/1000000)
        obs_y = float(decimal.Decimal(random.randrange(-118488471, -118485219))/1000000)
        pos = catalina.create_cartesian((obs_x, obs_y), catalina.ORIGIN_BOUND)
        obs_size = int(decimal.Decimal(random.randrange(10, 30))/1)
        obs = Motion_plan_state(pos[0], pos[1], size=obs_size)
        obstacles.append(obs)

        habitat_x = float(decimal.Decimal(random.randrange(33443758, 33445914))/1000000)
        habitat_y =  float(decimal.Decimal(random.randrange(-118488471, -118485219))/1000000)
        pos = catalina.create_cartesian((habitat_x, habitat_y), catalina.ORIGIN_BOUND)
        habitat_size = int(decimal.Decimal(random.randrange(50, 120))/1)
        habi = Motion_plan_state(pos[0], pos[1], size=habitat_size)
        habitats.append(habi)

    for obs in obstacles:
        test_a.append((obs.x, obs.y, obs.size))
    
    for habi in habitats:
        test_b.append((habi.x, habi.y, habi.size))

    print("\n", "obstacles = ", test_a, "habitats = ", test_b)
    return {"obstacles" : obstacles, "habitats" : habitats}

def iterate(times):
    weights = [0, 10, 10, 100]
    pathLenLimit = 200 # in meters 
    total_traj_time = 200 # in seconds

    start_cartesian = catalina.create_cartesian((33.446198, -118.486652), catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    obstacle_list = environ[0]
    boundary_list = environ[1]
    boat_list = environ[2]

    x_list = [1, 5, 10, 15, 20]
    y_list = []

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

    for complexity in x_list: # for each kind of environment 

        cost_list = []

        for sim in range(times): # run a number of simulations 

            obs_habi_dict = build_environ(complexity) # returns {"obstacles" : obstacles, "habitats" : habitats}
        
            final_obstacle_list = obstacle_list+boat_list+obs_habi_dict["obstacles"]
            final_habitat_list = obs_habi_dict["habitats"]

            astar_solver = astar(start, final_obstacle_list, boundary_list, final_habitat_list, {}, shark_dict, AUV_velocity=1) 
            final_path_mps = astar_solver.astar(pathLenLimit, weights, shark_dict)
            peri_boundary = cal_boundary_peri(astar_solver)

            traj_node = final_path_mps["node"]
            traj_mps = final_path_mps["path"]
            print ("\n", "Final trajectory : ", traj_mps)

            cost = get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, final_habitat_list, astar_solver.sharkGrid, weights)
            cost_list.append(cost)
            print ("\n", "cost list: ", cost_list)

        if len(cost_list) >= 1:   
            y_list.append(mean(cost_list))
            print ("\n","y values: ", y_list)

    return y_list

if __name__ == "__main__":
    y_list = iterate(times=10)
    print ("\n", "YLIST: ", y_list)

    

