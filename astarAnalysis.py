import math
import timeit
import statistics
import random
import decimal
import catalina
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost 
from path_planning.astar_fixLenSOG import astar, Node 
from shapely.geometry import Polygon
from path_planning.rrt_dubins import RRT, createSharkGrid
from motion_plan_state import Motion_plan_state
from habitatZones import main_shark_traj_function
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

def usable_obs(obstacle, start, obstacle_list):
    """
    Return True if the randomly generated obstacle is not on the start position

    Parameter:
        obstacle: a Motion_plan_state object
        start: a tuple of two elements 
    """
    # print("usable_obs called")
    dx = abs(start[0] - obstacle.x)
    dy = abs(start[1] - obstacle.y)

    dist = math.sqrt(dx**2 + dy**2) 
    if dist > obstacle.size: # check no overlap with the start position
        if no_overlap(obstacle, obstacle_list): # check no overlap with the previous obstacles 
            return True
        else:
            return False
    else:
        return False
    
def no_overlap(obstacle, obstacle_list):
    """
    Return True if the newly generated obstacle has no overlap with the previously generated obstacles in obstacle_list;
    Return false otherwise

    Parameter: 
	    obstacle: a Motion_plan_state object 
	    obstacle_list: a list of Motion_plan_state objects
    """
    # print("no_overlap called")
    if len(obstacle_list) != 0:
        for obs in obstacle_list:

            dx = obs.x - obstacle.x
            dy = obs.y - obstacle.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist <= obs.size + obstacle.size:
                return False	
            else:
                return True 
    else:
        return True

def build_habitats(complexity):
    """
    Generate the right number of randomized habitats corresponding to the complexity level

    Parameter:
        complexity: an integer represents the number of habitats generated
    """
    habitats = []
    test_a = []

    for i in range(complexity):
        habitat_x = float(decimal.Decimal(random.randrange(33443758, 33445914))/1000000)
        habitat_y =  float(decimal.Decimal(random.randrange(-118488471, -118485219))/1000000)
        pos = catalina.create_cartesian((habitat_x, habitat_y), catalina.ORIGIN_BOUND)
        habitat_size = int(decimal.Decimal(random.randrange(50, 120))/1)
        habi = Motion_plan_state(pos[0], pos[1], size=habitat_size)
        habitats.append(habi)

    for habi in habitats:
        test_a.append((habi.x, habi.y, habi.size))
    
    print("\n", "habitats = ", test_a)
    print("habitat account = ", len(habitats))
    return habitats

def build_obstacles(complexity, start):
    """
    Generate the right number of randomized obstacles corresponding to the complexity level

    Parameter:
        complexity: an integer represents the number of obstacles generated
    """ 
    obstacles = []
    test_b = []
    i = 0

    while i != complexity:
        obs_x = float(decimal.Decimal(random.randrange(33443758, 33445914))/1000000)
        obs_y = float(decimal.Decimal(random.randrange(-118488471, -118485219))/1000000)
        pos = catalina.create_cartesian((obs_x, obs_y), catalina.ORIGIN_BOUND)
        obs_size = int(decimal.Decimal(random.randrange(5, 50))/1)
        obs = Motion_plan_state(pos[0], pos[1], size=obs_size)
        # print ("obs generated: ", (obs.x, obs.y, obs.size))
        if usable_obs(obs, start, obstacles):
            # print("usable")
            obstacles.append(obs)
            i += 1

    for obs in obstacles:
        test_b.append((obs.x, obs.y, obs.size))

    print("\n", "obstacles = ", test_b)
    print("obstacle account = ", len(obstacles))
    return obstacles

def build_sharkDict(complexity):
    """
    Generate the right number of randomized shark trajectories corresponding to the complexity level

    Parameter:
        complexity: an integer represents the number of shark trajectories generated
    """  
    GRID_RANGE = 10
    DELTA_T = 2 # dt = 2s

    for shark_num in range(complexity):
        shark_dict = main_shark_traj_function(GRID_RANGE, shark_num+1, DELTA_T)
    
    return shark_dict

def build_environ(complexity, start): 
    """
    Generate the right number of randomized obstacles, habitats, and shark trajectories 
    corresponding to the complexity level

    Parameter:
        complexity: an integer; represents the number of obstacles and habitats
    """
    
    habitats = []
    obstacles = []
    shark_dict = {}

    test_a = []
    test_b = []
    GRID_RANGE = 10
    DELTA_T = 2 # dt = 2s

    for shark_num in range(complexity):
        shark_dict = main_shark_traj_function(GRID_RANGE, shark_num+1, DELTA_T)
        
    i = 0
    while i != complexity:
        obs_x = float(decimal.Decimal(random.randrange(33443758, 33445914))/1000000)
        obs_y = float(decimal.Decimal(random.randrange(-118488471, -118485219))/1000000)
        pos = catalina.create_cartesian((obs_x, obs_y), catalina.ORIGIN_BOUND)
        obs_size = int(decimal.Decimal(random.randrange(5, 50))/1)
        obs = Motion_plan_state(pos[0], pos[1], size=obs_size)
        # print ("obs generated: ", (obs.x, obs.y, obs.size))
        if usable_obs(obs, start, obstacles):
            # print("usable")
            obstacles.append(obs)
            i += 1

    for i in range(complexity):
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
    print ("Number of obstacles = ", len(obstacles), "Number of habitats = ", len(habitats))
    print ("Shark trajectories = ", shark_dict)
    return {"obstacles" : obstacles, "habitats" : habitats, "sharks" : shark_dict}

def habitats_vs_costTime(times, start, XLIST, shark_dict, weights, pathLenLimit, total_traj_time):
    """
    Keep the information of shark trajectories and obstacles constant to see the association between the number of habitats and cost

    Parameter:
        times: an integer represents the number of iterations
        start: a position tuple of two elements in the system of longtitudes and latitudes
        XLIST: a list of integers each of which signifies the complexity level(i.e. item account)
        shark_dict: a list of Motion_plan_state objects; represents multiple shark trajectories 
        weights: a list of four integers; we use [0, 10, 10, 100]
        total_traj_time: in seconds; set it to 200s 
    """

    COST = [] 
    TIME = []

    start_cartesian = catalina.create_cartesian(start, catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    obstacle_list = environ[0]+environ[2]
    boundary_list = environ[1]

    for complexity in XLIST: # for each kind of environment 

        cost_list = []
        time_list = []

        for sim in range(times): # run a number of simulations 

            habitat_list = build_habitats(complexity) # updating 
    
            astar_solver = astar(start, obstacle_list, boundary_list, habitat_list, {}, shark_dict, AUV_velocity=1) 
            start_time = timeit.timeit()
            final_path_mps = astar_solver.astar(pathLenLimit, weights, shark_dict)
            end_time = timeit.timeit()
            peri_boundary = cal_boundary_peri(astar_solver)

            traj_node = final_path_mps["node"]
            traj_mps = final_path_mps["path"]
            print ("\n", "Final trajectory : ", traj_mps)

            cost = get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, habitat_list, astar_solver.sharkGrid, weights)
            cost_list.append(cost)
            print ("\n", "cost list: ", cost_list)

            time_list.append(abs(end_time-start_time))
            print ("time list: ", time_list)

        if len(cost_list) >= 1:   
            COST.append(mean(cost_list))
            print ("\n","cost values: ", COST)
        if len(time_list) >= 1:
            TIME.append(mean(time_list))
            print("\n", "time values: ", TIME)

    return {"cost" : COST, "time" : TIME}

def obstacles_vs_costTime(times, start, XLIST, shark_dict, weights, pathLenLimit, total_traj_time):
    """
    Keep the information of shark trajectories and habitats constant to see the association between the number of obstacles and cost

    Parameter:
        times: an integer represents the number of iterations
        start: a position tuple of two elements in the system of longtitudes and latitudes
        XLIST: a list of integers each of which signifies the complexity level(i.e. item account)
        shark_dict: a list of Motion_plan_state objects; represents multiple shark trajectories 
        weights: a list of four integers; we use [0, 10, 10, 100]
        total_traj_time: in seconds; set it to 200s 
    """

    COST = [] 
    TIME = []

    start_cartesian = catalina.create_cartesian(start, catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    habitat_list = environ[3]
    boundary_list = environ[1]
    obstacle_list = environ[0]+environ[2] # updating

    for complexity in XLIST: # for each kind of environment 

        cost_list = []
        time_list = []

        for sim in range(times): # run a number of simulations 

            moreObs = build_obstacles(complexity, start) 
    
            astar_solver = astar(start, obstacle_list + moreObs, boundary_list, habitat_list, {}, shark_dict, AUV_velocity=1) 
            start_time = timeit.timeit()
            final_path_mps = astar_solver.astar(pathLenLimit, weights, shark_dict)
            end_time = timeit.timeit()
            peri_boundary = cal_boundary_peri(astar_solver)

            traj_node = final_path_mps["node"]
            traj_mps = final_path_mps["path"]
            print ("\n", "Final trajectory : ", traj_mps)

            cost = get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, habitat_list, astar_solver.sharkGrid, weights)
            cost_list.append(cost)
            print ("\n", "cost list: ", cost_list)

            time_list.append(abs(end_time-start_time))
            print ("\n", "time list: ", time_list)

        if len(cost_list) >= 1:   
            COST.append(mean(cost_list))
            print ("\n","cost values: ", COST)
        if len(time_list) >= 1:
            TIME.append(mean(time_list))
            print("\n", "time values: ", TIME)

    return {"cost" : COST, "time" : TIME}

def sharks_vs_costTime(times, start, XLIST, weights, pathLenLimit, total_traj_time):
    """
    Keep the information of shark trajectories and habitats constant to see the association between the number of obstacles and cost

    Parameter:
        times: an integer represents the number of iterations
        start: a position tuple of two elements in the system of longtitudes and latitudes
        XLIST: a list of integers each of which signifies the complexity level(i.e. item account)
        shark_dict: a list of Motion_plan_state objects; represents multiple shark trajectories 
        weights: a list of four integers; we use [0, 10, 10, 100]
        total_traj_time: in seconds; set it to 200s 
    """

    COST = [] 
    TIME = []

    start_cartesian = catalina.create_cartesian(start, catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    habitat_list = environ[3]
    boundary_list = environ[1]
    obstacle_list = environ[0]+environ[2]
    
    for complexity in XLIST: # for each kind of environment 

        cost_list = []
        time_list = []

        for sim in range(times): # run a number of simulations 

            shark_dict = build_sharkDict(complexity) 
    
            astar_solver = astar(start, obstacle_list, boundary_list, habitat_list, {}, shark_dict, AUV_velocity=1) 
            start_time = timeit.timeit()
            final_path_mps = astar_solver.astar(pathLenLimit, weights, shark_dict)
            end_time = timeit.timeit()
            peri_boundary = cal_boundary_peri(astar_solver)

            traj_node = final_path_mps["node"]
            traj_mps = final_path_mps["path"]
            print ("\n", "Final trajectory : ", traj_mps)

            cost = get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, habitat_list, astar_solver.sharkGrid, weights)
            cost_list.append(cost)
            print ("\n", "cost list: ", cost_list)

            time_list.append(abs(end_time-start_time))
            print ("\n", "time list: ", time_list)

        if len(cost_list) >= 1:   
            COST.append(mean(cost_list))
            print ("\n","cost values: ", COST)
        if len(time_list) >= 1:
            TIME.append(mean(time_list))
            print("\n", "time values: ", TIME)

    return {"cost" : COST, "time" : TIME}

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

    x_list = [1, 3, 5, 7, 9]
    y_list = [] # holds averaged costs
    z_list = [] # holds averaged planning time

    
    '''shark_dict = {1: [Motion_plan_state(-120 + (0.3 * i), -60 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    2: [Motion_plan_state(-65 - (0.3 * i), -50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    3: [Motion_plan_state(-110 + (0.3 * i), -40 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    4: [Motion_plan_state(-105 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    5: [Motion_plan_state(-120 + (0.3 * i), -50 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    6: [Motion_plan_state(-85 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    7: [Motion_plan_state(-270 + (0.3 * i), 50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    8: [Motion_plan_state(-250 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    9: [Motion_plan_state(-260 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    10: [Motion_plan_state(-275 + (0.3 * i), 80 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)]} '''

    for complexity in x_list: # for each kind of environment 

        cost_list = []
        time_list = []

        for sim in range(times): # run a number of simulations 

            environ_dict = build_environ(complexity, start) # returns {"obstacles" : obstacles, "habitats" : habitats}
        
            final_obstacle_list = obstacle_list+boat_list+environ_dict["obstacles"]
            final_habitat_list = environ_dict["habitats"]
            shark_dict = environ_dict["sharks"]

            astar_solver = astar(start, final_obstacle_list, boundary_list, final_habitat_list, {}, shark_dict, AUV_velocity=1) 
            start_time = timeit.timeit()
            final_path_mps = astar_solver.astar(pathLenLimit, weights, shark_dict)
            end_time = timeit.timeit()
            peri_boundary = cal_boundary_peri(astar_solver)

            traj_node = final_path_mps["node"]
            traj_mps = final_path_mps["path"]
            print ("\n", "Final trajectory : ", traj_mps)

            cost = get_cost(traj_node, traj_mps, peri_boundary, total_traj_time, final_habitat_list, astar_solver.sharkGrid, weights)
            cost_list.append(cost)
            print ("\n", "cost list: ", cost_list)

            time_list.append(abs(end_time-start_time))
            print ("\n", "time list: ", time_list)

        if len(cost_list) >= 1:   
            y_list.append(mean(cost_list))
            print ("\n","y values: ", y_list)
        if len(time_list) >= 1:
            z_list.append(mean(time_list))
            print("\n", "z values: ", z_list)

    return {"Y" : y_list, "Z" : z_list}

if __name__ == "__main__":

    # shark_dict = {1: [Motion_plan_state(-120 + (0.3 * i), -60 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    # 2: [Motion_plan_state(-65 - (0.3 * i), -50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    # 3: [Motion_plan_state(-110 + (0.3 * i), -40 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    # 4: [Motion_plan_state(-105 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    # 5: [Motion_plan_state(-120 + (0.3 * i), -50 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    # 6: [Motion_plan_state(-85 - (0.3 * i), -55 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    # 7: [Motion_plan_state(-270 + (0.3 * i), 50 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    # 8: [Motion_plan_state(-250 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)],
    # 9: [Motion_plan_state(-260 - (0.3 * i), 75 + (0.3 * i), traj_time_stamp=i) for i in range(1,201)], 
    # 10: [Motion_plan_state(-275 + (0.3 * i), 80 - (0.3 * i), traj_time_stamp=i) for i in range(1,201)]} 

    iterations = 10 
    pathLenLimit = 200 
    total_traj_time = 200
    start = (33.446198, -118.486652)
    weights = [0, 10, 10, 100]
    XLIST = [1, 3, 5, 7, 9]

    output = sharks_vs_costTime(iterations, start, XLIST, weights, pathLenLimit, total_traj_time)
    cost = output["cost"]
    time = output["time"]

    print("\n", "CostList = ", cost, "TimeList = ", time)

    # output = iterate(times=10)
    # y_list = output["Y"]
    # z_list = output["Z"]
    # print ("\n", "YLIST: ", y_list, "ZLIST: ", z_list)

    

