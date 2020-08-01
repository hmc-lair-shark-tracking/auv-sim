import math
import geopy.distance
import random
import timeit
from motion_plan_state import Motion_plan_state
import matplotlib.pyplot as plt
import numpy as np
import catalina

from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon 
from catalina import create_cartesian
from cost import Cost

def euclidean_dist(point1, point2):
    """
    Calculate the distance square between two points

    Parameter:
        point1 - a position tuple: (x, y)
        point2 - a position tuple: (x, y)
    """

    dx = abs(point1[0]-point2[0])
    dy = abs(point1[1]-point2[1])

    return math.sqrt(dx*dx+dy*dy)

class Node: 
    # a node in the graph

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0 
        self.h = 0
        self.f = 0  

        self.cost = 0 # each node has a cost value computed through def cost_of_edge

class astar:

    def __init__(self, start, goal, obs_lst, boundary):
        self.path = [] # a list of motion plan state 
        self.start = start
        self.goal = goal
        self.obstacle_list = obs_lst
        self.boundary_list = boundary
        # self.habitats = []
        
    def euclidean_dist(self, point1, point2): # point is a position tuple (x, y)
        """
        Calculate the distance square between two points

        Parameter:
            point1 - a position tuple: (x, y)
            point2 - a position tuple: (x, y)
        """

        dx = abs(point1[0]-point2[0])
        dy = abs(point1[1]-point2[1])

        return dx*dx+dy*dy
    
    def get_distance_angle(self, start_mps, end_mps):
        """
        Calculate the distance and angle between two points

        Parameter:
            start_mps - a Motion_plan_state object
            end_mps - a Motion_plan_state object
        """

        dx = end_mps.x-start_mps.x
        dy = end_mps.y-start_mps.y
        
        dist = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy,dx)
        return dist, theta

    def same_side(self, p1, p2, a, b):
        cp1 = np.cross(np.asarray(b)-np.asarray(a), np.asarray(p1)-np.asarray(a))
        cp2 = np.cross(np.asarray(b)-np.asarray(a), np.asarray(p2)-np.asarray(a))
        if np.dot(cp1, cp2) >= 0:
            return True
        else:
            return False
    
    def point_in_triangle(self, p, a, b, c):
        if self.same_side(p, a, b, c) and self.same_side(p, b, a, c) and self.same_side(p, c, a, b):
            return True
        else:
            return False
    
    def check_boundary(self, boundary_list, position):
        """
        Check if the given position is within the given booundry that is 
        """
        poly_list = []  
    
        for corner in boundary_list: 
            poly_list.append([corner.x, corner.y])

        centroid = Polygon(poly_list).centroid.coords
        for index in range(len(poly_list)):
            if index != len(poly_list)-1:
                if self.point_in_triangle(position, poly_list[index], poly_list[index+1], centroid):
                    return True 
            else:
                if self.point_in_triangle(position, poly_list[len(poly_list)-1], poly_list[0], centroid):
                    return True
        return False # not within boundary  
        
    def check_collision(self, position, obstacleList): # obstacleList lists of motion plan state 
        """
        Check if the current position is collision-free

        Parameter:
            position - a tuple with two elements, x and y coordinates
            obstacleList - a list of Motion_plan_state objects
        """

        position_mps = Motion_plan_state(position[0], position[1])

        for obstacle in obstacleList:
            d, _ = self.get_distance_angle(obstacle, position_mps)
            if d <= obstacle.size:
                return False # collision 

        return True 

    def curr_neighbors(self, current_node, boundary_list): 

        """
        Return a list of position tuples that are close to the current point

        Parameter:
            current_node: a Node object 
        """
        adjacent_squares = [(0, -10), (0, 10), (-10, 0), (10, 0), (-10, -10), (-10, 10), (10, -10), (10, 10)]
            
        current_neighbors = []

        for new_position in adjacent_squares:
            
            node_position = (current_node.position[0]+new_position[0], current_node.position[1]+new_position[1])

            # check if it's within the boundary

            if self.check_boundary(boundary_list, node_position):
                current_neighbors.append(node_position)

        return current_neighbors  
    
    def near_goal(self, coords, goal):
        dist = self.euclidean_dist(coords, goal)
        if dist <= 100: 
            return True # near goal if the current position is 10 meter or less from the goal
        else:
            False

    def habitats_time_spent(self, current_node):
        """
        Find the approximate time spent in habitats if the current node is within the habitat(s)

        Parameter:
            current_node: a position tuple of two elements (x,y)
            habitats: a list of motion_plan_state
        """
        dist_in_habitats = 0
        velocity = 1 # velocity of exploration in m/s

        for habi in catalina.HABITATS:
            # print ('habitat size: ', habi.size)
            pos_habi = create_cartesian((habi.x, habi.y), catalina.ORIGIN_BOUND)
            dist, _ = self.get_distance_angle(Motion_plan_state(pos_habi[0], pos_habi[1], size=habi.size), Motion_plan_state(current_node.position[0], current_node.position[1]))
            # print ('distance from habitat: ', dist)
            if dist <= habi.size:
                dist_in_habitats += 10
                
        return dist_in_habitats/velocity
    
    def check_cover_habitats(self, current_node, habitat_open_list, habitat_closed_list):
        for index, item in enumerate(habitat_open_list):
            dist = math.sqrt((current_node.position[0]-item.x) **2 + (current_node.position[1]-item.y) **2)
            if dist <= item.size: # current_node covers a habitat
                habitat_open_list.pop(index)
                habitat_closed_list.append(item)

        return (habitat_open_list, habitat_closed_list)

    def create_grid_map(self, current_node, neighbor):

        habitat_list = []
        constant = 1

        for habi in catalina.HABITATS:
            pos = catalina.create_cartesian((habi.x, habi.y), catalina.ORIGIN_BOUND)
            habitat_list.append(Motion_plan_state(pos[0], pos[1], size=habi.size))
        print ('habitats: ', habitat_list)

        target_habitat = habitat_list[0]
        min_distance = euclidean_dist((target_habitat.x, target_habitat.y), current_node.position)

        for habi in habitat_list:
            dist = euclidean_dist((habi.x, habi.y), current_node.position)
            if dist < min_distance:
                target_habitat = habi
        print ('target: ', target_habitat)
        
        # compute Nj 
        vector_1 = [neighbor[0]-current_node.position[0], neighbor[1]-current_node.position[1]]
        print ('vector_1: ', vector_1)
        vector_2 = [target_habitat.x-current_node.position[0], target_habitat.y-current_node.position[1]]
        print ('vector_2: ', vector_2)
        unit_vector_1 = vector_1/np.linalg.norm(vector_1)
        unit_vector_2 = vector_2/np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        print ('angle: ', angle)
        Nj = math.cos(angle)
        
        # compute Gj 
        for item in habitat_list:
            dist = math.sqrt((current_node.position[0]-item.x) **2 + (current_node.position[1]-item.y) **2)
            if dist <= item.size: # current_node covers a habitat
                Gj = 1
            else:
                Gj = 0

        # compute grid value bj        
        bj = constant * Nj + Gj 

        return (bj) 

    def astar(self, habitat_list, obs_lst, boundary_list, start, goal, weights): 
        """
        Find the optimal path from start to goal avoiding given obstacles 

        Parameter: 
            obs_lst - a list of motion_plan_state objects that represent obstacles 
            start - a tuple of two elements: x and y coordinates
            goal - a tuple of two elements: x and y coordinates
        """

        habitats_time_spent = 0
        cal_cost = Cost()
        path_length = 0

        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, goal)
        end_node.g = end_node.h = end_node.f = 0

        dynamic_path = [] # append new node as a* search goes

        cost = []
        open_list = [] # hold neighbors of the expanded nodes
        closed_list = [] # hold all the exapnded nodes
        habitat_open_list = habitat_list # hold haibitats that have not been explored 
        habitat_closed_list = [] # hold habitats that have been explored 

        open_list.append(start_node)

        while len(open_list) > 0:
            current_node = open_list[0] # initialize the current node
            current_index = 0

            for index, item in enumerate(open_list): # find the current node with the smallest f(f=g+h)
                if item.f  < current_node.f:
                    current_node = item
                    current_index = index
            
            open_list.pop(current_index)
               
            closed_list.append(current_node)
            dynamic_path.append(current_node.position)
            print ("dynamic path: ", dynamic_path)
            # print ('dynamic_path: ', dynamic_path)
   
            if self.near_goal(current_node.position, end_node.position):
                path = []
                current = current_node

                while current is not None: # backtracking to find the path 
                    path.append(current.position)
                    cost.append(current.cost)
                    # if current.parent is not None:  
                    #     current.cost = current.parent.cost + cal_cost.cost_of_edge(current, path, habitat_open_list, habitat_closed_list, weights=[1, 1, 1])
                    path_length += 10
                    habitats_time_spent += self.habitats_time_spent(current)
                    # print ("\n", "habitats_time_spent: ", habitats_time_spent)
                    current = current.parent
    
                path_mps = [] 
        
                for point in path:
                    mps = Motion_plan_state(point[0], point[1])
                    path_mps.append(mps)

                # print ("\n", 'actual path length: ', path_length)
                # print ("\n", 'time spent in habitats: ', habitats_time_spent)
                # print ("\n", "cost list: ", cost)
                # print ("\n", 'cost list length: ', len(cost))
                # print ("\n", 'path list length: ', len(path_mps))
                return ([path_mps[::-1], cost])

            # find close neighbors of the current node
            
            current_neighbors = self.curr_neighbors(current_node, boundary_list)
            print ("\n", "current neighbors: ", current_neighbors)

            # make current neighbors Nodes

            children = []
            grid = [] # holds a list of tuples (neighbor, grid value)

            for neighbor in current_neighbors: # create new node if the neighbor is collision-free
                if self.check_collision(neighbor, obs_lst):
                    """
                    SELECTION 
                    """ 
                    grid_val = self.create_grid_map(current_node, neighbor)   
                    grid.append((neighbor, grid_val))
                    print ("\n", "grid value: ", grid_val)

            print ("\n", 'grid list: ', grid)
            # remove two elements with the lowest grid values 
            grid.remove(min(grid)) 
            print ("selected grid list: ", grid)
            # grid.remove(min(grid))

            # append the selected neighbors to children 
            for neighbor in grid: 
                new_node = Node(current_node, neighbor[0])
                children.append(new_node)
                # update habitat_open_list and habitat_closed_list
                result = self.check_cover_habitats(new_node, habitat_open_list, habitat_closed_list)
                habitat_open_list = result[0]
                # print ("habitat_open_list: ", habitat_open_list)
                habitat_closed_list = result[1]
                # print ("habitat_closed_list: ", habitat_closed_list)

            for child in children: 
                if child in closed_list:
                    continue
                
                # dist_child_current = self.euclidean_dist(child.position, current_node.position)
                if child.parent is not None:
                    result = cal_cost.cost_of_edge(child, dynamic_path, habitat_open_list, habitat_closed_list, weights=weights)
                    child.cost = child.parent.cost + result[0]
                # child.g = current_node.g + dist_child_current
                child.g = current_node.cost 
                child.h = weights[0] * self.euclidean_dist(child.position, goal) - weights[1] * result[1] # result=(cost, d_2)
                child.f = child.g + child.h

                # check if child exists in the open list and have bigger g 
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                open_list.append(child)

def main():

    start = create_cartesian((33.444686, -118.484716), catalina.ORIGIN_BOUND)
    print("start: ", start)
     
    # print ('start: ', start)
    # print ('goal: ', goal)

    obstacle_list = []
    boundary_list = []
    habitat_list = []
    goal_list =[]
    weights = [0,10,10]

    final_cost = []

    for obs in catalina.OBSTACLES:
        pos = create_cartesian((obs.x, obs.y), catalina.ORIGIN_BOUND)
        obstacle_list.append(Motion_plan_state(pos[0], pos[1], size=obs.size))

    for b in catalina.BOUNDARIES:
        pos = create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
        boundary_list.append(Motion_plan_state(pos[0], pos[1]))

    for habi in catalina.HABITATS:
        pos = catalina.create_cartesian((habi.x, habi.y), catalina.ORIGIN_BOUND)
        habitat_list.append(Motion_plan_state(pos[0], pos[1], size=habi.size))

    goal1 = create_cartesian((33.445779, -118.486976), catalina.ORIGIN_BOUND)
    print ("goal: ", goal1)
    astar_solver = astar(start, goal1, obstacle_list, boundary_list) 
    final_path_mps = astar_solver.astar(habitat_list, obstacle_list, boundary_list, start, goal1, weights)
    print ("\n", "final cost: ",  final_path_mps[1][0])
    print ("\n", "final path: ", final_path_mps[0])
 
if __name__ == "__main__":
    main()
