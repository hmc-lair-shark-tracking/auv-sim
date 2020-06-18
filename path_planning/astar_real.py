import math
import geopy.distance
import random
import timeit
from motion_plan_state import Motion_plan_state
import matplotlib.pyplot as plt
import numpy as np
from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon 
from catalina import create_cartesian
import catalina

class Node: 
    # a node in the graph

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0 
        self.h = 0
        self.f = 0  

class astar:

    def __init__(self, start, goal, obs_lst, boundary):
        self.path = [] # a list of motion plan state 
        self.start = start
        self.goal = goal
        self.obstacle_list = obs_lst
        self.boundary_list = boundary
        
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

    def astar(self, obs_lst, boundary_list): 
        """
        Find the optimal path from start to goal avoiding given obstacles 

        Parameter: 
            obs_lst - a list of motion_plan_state objects that represent obstacles 
            start - a tuple of two elements: x and y coordinates
            goal - a tuple of two elements: x and y coordinates
        """

        start_node = Node(None, self.start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, self.goal)
        end_node.g = end_node.h = end_node.f = 0

        open_list = [] # hold neighbors of the expanded nodes
        closed_list = [] # hold all the exapnded nodes 

        open_list.append(start_node)

        while len(open_list) > 0:
            current_node = open_list[0] # initialize the current node
            current_index = 0

            for index, item in enumerate(open_list): # find the current node with the smallest f
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            
            open_list.pop(current_index)

            closed_list.append(current_node)
   
            if self.near_goal(current_node.position, end_node.position):
                path = []
                current = current_node

                while current is not None: # backtracking to find the path 
                    path.append(current.position)
                    current = current.parent
                
                path_mps = [] 
        
                for point in path:
                    mps = Motion_plan_state(point[0], point[1])
                    path_mps.append(mps)
    
                return path_mps[::-1]

            # find close neighbors of the current node
            
            current_neighbors = self.curr_neighbors(current_node, boundary_list)

            # make current neighbors Nodes

            children = []

            for neighbor in current_neighbors: # create new node if the neighbor is collision-free
                if self.check_collision(neighbor, obs_lst):
                    new_node = Node(current_node, neighbor)
                    children.append(new_node)

            for child in children: 
                if child in closed_list:
                    continue
                
                dist_child_current = self.euclidean_dist(child.position, current_node.position)

                child.g = current_node.g + dist_child_current
                child.h = self.euclidean_dist(child.position, self.goal) # distance from child to end goal
                child.f = child.g + child.h

                # check if child exists in the open list and have bigger g 
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                open_list.append(child)

def main():

    ### TEST COST FUNCTION


    ### TEST FINAL PATH 
    '''
    start = create_cartesian(catalina.START, catalina.ORIGIN_BOUND)
    goal = create_cartesian(catalina.GOAL, catalina.ORIGIN_BOUND)
 
    print ('start: ', start)
    
    print ('goal: ', goal)

    obstacles = catalina.OBSTACLES

    boundaries = catalina.BOUNDARIES

    obstacle_list = []
    boundary_list = []

    astar_solver = astar(start, goal, obstacle_list, boundary_list)

    for obs in obstacles:
        pos = create_cartesian((obs.x, obs.y), catalina.ORIGIN_OBS)
        obstacle_list.append(Motion_plan_state(pos[0], pos[1], size=obs.size))

    for b in boundaries:
        pos = create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
        boundary_list.append(Motion_plan_state(pos[0], pos[1]))

    final_path_mps = astar_solver.astar(obstacle_list, boundary_list, start, goal)
    print (final_path_mps)

    '''

if __name__ == "__main__":
    main()