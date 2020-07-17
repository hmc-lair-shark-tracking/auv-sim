import math
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np

#from rrt_dubins import RRT
from motion_plan_state import Motion_plan_state

class Cost:
    #class for building cost function in path planning

    def __init__(self):
        self.cost = 0

    def test_cost_func(self, path, length, bonus_area, weights=[1,0]):
        '''
        a testing cost function that added bonus to path passing through certain area in the configuration space
        we want to find a path minimizing path length while maximizing the number of motion_plan_states in the bonus area

        path: a list of motion_plan_states
        length: the length of current path
        bonus_area: for each motion_plan_state in the bonus area, some bonus will be added to the cost function
        weights: a list of weight for cost function
        '''
        #cost function = w1 * length - w2 * num(mps in bonus area)
        w1 = weights[0]
        w2 = weights[1]
        cost = w1 * length
        
        for mps in path:
            dist = math.sqrt((bonus_area[0].x-mps.x)**2 + (bonus_area[0].y-mps.y)**2)
            if dist <= bonus_area[0].size:
                cost -= w2

        return cost
    
    def habitat_num_cost_func(self, path, length, habitats, weights=[1,1]):
        '''
        cost function for habitat exploration, we want to find a path minimizing path length 
        while maximizing the number of habitats visited
        cost function = w1 * length - w2 * number of habitats visited

        path: a list of motion_plan_states
        length: the length of current path
        habitats: a list of habitat areas represented as motion_plan_state
        '''
        #set the weight for each term in cost function
        w1 = weights[0]
        w2 = weights[1]

        #number of habitats visited
        count = 0
        
        for habitat in habitats:
            for mps in path:
                dist = math.sqrt((habitat.x-mps.x) **2 + (habitat.y-mps.y) **2)
                if dist <= habitat.size:
                    count +=1
                    break
        
        cost = w1 * length - w2 * count

        return cost

    def cost_of_edge (self, new_node, habitat_open_list, habitat_closed_list, weights):

        """
        Calculate the cost of the new edge constructed when the new node is appended to the path list. 
        The cost of edge accumulates to form the ultimate cost of the particular path under the specified condition.

        Parameter: 
            new_node: a Node object represents the next node to connect to the existing path 
            current_path: a list of Motion_plan_state objects, dynamically changing as more Node objects added to the existing path
            habitat_open_list: a list of Motion_plan_state objects holds habitats that have been covered 
            habitat_closed_list: a list of Motion_plan_state objects holds new habitats that have not been covered 
            weights: a list of three numbers 
        """

        #set the weight for each term in cost function
        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        d_2 = 0 
        d_3 = 0 

        # set d_2
        for habi in habitat_open_list+habitat_closed_list:
            dist = math.sqrt((new_node.position[0]-habi.x) **2 + (new_node.position[1]-habi.y) **2)
            if dist <= habi.size:
                d_2 = 1
                
        # set d_3
        for habi in habitat_closed_list:
            dist = math.sqrt((new_node.position[0]-habi.x) **2 + (new_node.position[1]-habi.y) **2)
            if dist <= habi.size:
                d_3 = 1
        
        cost_of_edge = - w2 * d_2 - w3 * d_3

        return ([cost_of_edge, d_2, d_3])


    def habitat_time_cost_func(self, path, length, habitats, dist, weights=[1,-1,-1]):
        '''
        cost function for habitat exploration, we want to find a path minimizing path length 
        while maximizing the time spent in different habitats visited
        cost function = w1 * length - w2 * number of habitats visited - w3 * time spent in different habitats
       
        path: a list of motion_plan_states
        length: the length of current path
        habitats: a list of habitat areas represented as motion_plan_state

        output: 
        cost: [total cost, [cost for w1, cost for w2, ...]]
        '''
        #set the weight for each term in cost function
        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        cost = [0 for _ in range(len(weights))]
        #normalize the cost for path length
        cost[0] = w1 * length / dist
        
        visited = {} #dictionary of visited habitats
        for i in range(len(habitats)):
            visited[i+1] = False #visited initialized to be False for all habitat

        for i in range(len(habitats)):
            for mps in path:
                dist = math.sqrt((habitats[i].x-mps.x) **2 + (habitats[i].y-mps.y) **2)
                if dist <= habitats[i].size:
                    visited[i+1] = True
                    cost[2] += w3
                    
        #normalize the cost for time spent in habitats
        cost[2] = cost[2] / (0.5 * dist)
    
        count = 0 #number of habitats visited
        for i in range(1, len(visited)+1):
            if visited[i] == True:
                count += 1
        
        #normalize the cost for number of habitats visited
        cost[1] = w2 * count / len(habitats) 

        return [sum(cost), cost]
    
    def habitat_shark_cost_func(self, path, length, peri, total_traj_time, habitats, shark_dict, weight):
        '''
        cost function for habitat exploration and shark tracking
        we want to find a path minimizing path length, maximizing the time spent in different habitats visited,
            maximizing the number of sharks in the range of sonar detection of AUV
        
        cost function = w1 * length - w2 * number of habitats visited - w3 * time spent in different habitats 
                        - w4 * number of sharks in range

        path: current path, represented as a list of motion_plan_states
        length: the length of current path
        peri: perimeter of boundary, a normalization factor for path length
        habitats: a list of habitat areas represented as motion_plan_states
        shark_dict: a dictionary representing different shark trajectory/position

        output: 
        cost: [total cost, [cost for w1, cost for w2, ...]]
        '''
        #set the weight for each term in cost function
        w1 = weight[0]
        w2 = weight[1]
        w3 = weight[2]
        w4 = weight[3]

        cost = [0 for _ in range(len(weight))]
        #normalize the cost for path length
        cost[0] = w1 * length / peri
        
        visited = {} #dictionary of visited habitats
        for i in range(len(habitats)):
            visited[i+1] = False #visited initialized to be False for all habitat
        
        for mps in path:
            found = False
            for time_bin in shark_dict:
                if mps.traj_time_stamp >= time_bin[0] and mps.traj_time_stamp <= time_bin[1]:
                    temp_time = time_bin
                    found = True
                    break
            if not found:
                continue
            sharkGrid = shark_dict[temp_time]
            for cell_bound, prob in sharkGrid.items():
                if mps.x >= cell_bound[0] and mps.x <= cell_bound[2] and mps.y >= cell_bound[1] and mps.x <= cell_bound[3]:
                    cost[3] += w4 * prob
                    break

            for i in range(len(habitats)):
                dist = math.sqrt((habitats[i].x-mps.x) **2 + (habitats[i].y-mps.y) **2)
                if dist <= habitats[i].size:
                    visited[i+1] = True
                    cost[2] += w3
                    break
        
        #normalize the cost for time spent in habitats
        cost[2] = cost[2] / total_traj_time
        cost[3] = cost[3] / total_traj_time
    
        count = 0 #number of habitats visited
        for i in range(1, len(visited)+1):
            if visited[i] == True:
                count += 1
        
        #normalize the cost for number of habitats visited
        if len(habitats) != 0:
            cost[1] = w2 * count / len(habitats)

        return [sum(cost), cost]