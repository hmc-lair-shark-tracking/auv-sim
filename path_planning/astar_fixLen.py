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
        self.pathLen = 0 # dynamically changing as a node is appended to the existing length
        self.time_stamp = 0

class astar:

    def __init__(self, start, obstacleList, boundaryList, habitatList):
        self.path = [] # a list of motion plan state 
        self.start = start
        self.obstacle_list = obstacleList
        self.boundary_list = boundaryList
        self.habitat_list = habitatList
        self.visited_nodes = np.zeros([600, 600])
        self.velocity = 1 
        
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
    
    def within_bounds(self, boundary_list, position):
        """
        Check if the given position is within the given booundry expressed as a polygon
        Paramter: 
            boundary_list: a list of Motion_plan_state objects that define the corners of the region of interest
            position: a tuple of two elements (x, y) to test whether it's within the boundary or not 
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
        
    def collision_free(self, position, obstacleList): # obstacleList lists of motion plan state 
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

    def Walkable(self, check_point, current_point, obstacleList):
        """
        Return true if there's no obstacle from check point to current point => walkable;
        return false otherwise

        Parameter:
            check_point: a Motion_plan_state object
            current_point: a Motion_plan_state object
        """

        # print ("\n", "Walkable called")
        start_x = check_point.x
        start_y = check_point.y

        step_x = int(abs(current_point.x - check_point.x)/5)
        step_y = int(abs(current_point.y - check_point.y)/5)
    
        while start_x <= current_point.x and start_y <= current_point.y:
            intermediate = (start_x, start_y)
            # print ("intermediate position: ", intermediate)
            start_x += step_x
            start_y += step_y

            if not self.collision_free(intermediate, obstacleList): # if collision happens
                return False

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

            if self.within_bounds(boundary_list, node_position):
                current_neighbors.append(node_position)

        return current_neighbors  

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
    
            pos_habi = create_cartesian((habi.x, habi.y), catalina.ORIGIN_BOUND)
            dist, _ = self.get_distance_angle(Motion_plan_state(pos_habi[0], pos_habi[1], size=habi.size), Motion_plan_state(current_node.position[0], current_node.position[1]))
            
            if dist <= habi.size:
                dist_in_habitats += 10
                
        return dist_in_habitats/velocity
    
    def update_habitat_coverage(self, current_node, habitat_open_list, habitat_closed_list):
        """
        Check if the current node covers a habitat (either explored or unexplored);
        then update the habitat_open_list that holds all unexplored habitats
        and update the habitat_closed_list that holds all explored habitats 
        Parameter:
            current_node: a Node object 
            habitat_open_list: a list of Motion_plan_state objects
            habitat_closed_list: a list of Motion_plan_state objects
        """
        for index, item in enumerate(habitat_open_list):
            dist = math.sqrt((current_node.position[0]-item.x) **2 + (current_node.position[1]-item.y) **2)
            if dist <= item.size: # current_node covers a habitat
                habitat_open_list.pop(index)
                habitat_closed_list.append(item)

        return (habitat_open_list, habitat_closed_list)
    
    def inside_habitats(self, mps, habitats):
        """
        Return True if the location mps is inside any of the habitats in habitat_list;
        return False otherwise

        Parameter:
            mps: a Motion_plan_state object; the location to check
            habitat_list: a list of Motion_plan_state objects
        """
    
        for habitat in habitats:
            dist = euclidean_dist((habitat.x, habitat.y), (mps.x, mps. y))
           
            if dist <= habitat.size:
                return True

        return False
        
    def create_grid_map(self, current_node, neighbor):
        """
        Find the grid value of the current node's neighbor;
        the magnitude of the grid value is influenced by the angle of the current node to the nearest habitat 
        and whether the current_node covers a habitat
        Parameter: 
            current_node: a Node object
            neighbor: a position tuple of two elements (x_in_meters, y_in_meters)
        """

        habitat_list = []
        constant = 1

        for habi in catalina.HABITATS:
            pos = catalina.create_cartesian((habi.x, habi.y), catalina.ORIGIN_BOUND)
            habitat_list.append(Motion_plan_state(pos[0], pos[1], size=habi.size))

        target_habitat = habitat_list[0]
        min_distance = euclidean_dist((target_habitat.x, target_habitat.y), current_node.position)

        for habi in habitat_list:
            dist = euclidean_dist((habi.x, habi.y), current_node.position)
            if dist < min_distance:
                target_habitat = habi
        
        # compute Nj 
        vector_1 = [neighbor[0]-current_node.position[0], neighbor[1]-current_node.position[1]]
        vector_2 = [target_habitat.x-current_node.position[0], target_habitat.y-current_node.position[1]]
     
        unit_vector_1 = vector_1/np.linalg.norm(vector_1)
        unit_vector_2 = vector_2/np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
       
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
    
    def sort_open_list(self, open_list):
        """
        Sort open_list in ascending order based on the f value of the node
        Parameter:
            open_list: a list of Node objects that are candidates of the next node to add to the path 
        """

        for i in range(len(open_list)):
            cursor = open_list[i] 
            pos = i
        
            while pos > 0 and open_list[pos - 1].f > cursor.f:
                # Swap the number down the list
                open_list[pos] = open_list[pos - 1]
                pos = pos - 1
            # Break and do the final swap
            open_list[pos] = cursor

        return open_list
    
    def get_indices(self, x_in_meters, y_in_meters):
        """
        Convert x, y in the coordinate system of the catalina environment to the coordinate system of the visited nodes
        in order to have these two positions correspond to each other 
        Parameter: 
            x_in_meters: a decimal
            y_in_meters: a decimal
        """
        x_pos = int(x_in_meters + 500)
        y_pos = int(y_in_meters + 200)

        return (x_pos, y_pos)

    def smoothPath(self, trajectory, haibitats): 
        """
        Return a smoothed trajectory after eliminating its intermediate waypoints

        Parameter: 
            trajectory: a list of Motion_plan_state objects
            obstacleList: a list of Motion_plan_state objects 
        """

        index = 0 
        checkPoint = trajectory[index] # starting point of the path
        index += 1
        currentPoint = trajectory[index] # next point in path
        smoothTraj = trajectory[:] # holds smoothed trajectory 
        
        while index < len(trajectory)-1:

            print ("\n", "index: ", index)

            if self.Walkable(checkPoint, currentPoint, self.obstacle_list): # if no obstacle in between two points
                
                inside_habitats = self.inside_habitats(currentPoint, haibitats)
                print ("\n", "currentPoint inside habitats? ", inside_habitats)
                print ("\n", "currentPoint: ", (currentPoint.x, currentPoint.y))
           
                if not inside_habitats: # if currentPoint is NOT within any of the habitats => removable
                    temp = currentPoint
                    index += 1 
                    currentPoint = trajectory[index]                    
                    smoothTraj.remove(temp)
                    
                    print ("\n", "Eliminate: ", (temp.x, temp.y)) 
                    print ("\n", "Walkable traj: ", smoothTraj)
                else: 
                    index += 1
                    currentPoint = trajectory[index]
                    
            else:
                checkPoint = currentPoint
                index += 1
                currentPoint = trajectory[index]
                print ("\n", "currentPoint NOT Walkable: ", (currentPoint.x, currentPoint.y))

        return smoothTraj

    def astar(self, habitat_list, obs_lst, boundary_list, start, pathLenLimit, weights): 
        """
        Find the optimal path from start to goal avoiding given obstacles 
        Parameter: 
            obs_lst - a list of motion_plan_state objects that represent obstacles 
            start - a tuple of two elements: x and y coordinates
            goal - a tuple of two elements: x and y coordinates
        """
        
        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        habitats_time_spent = 0
        cal_cost = Cost()

        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0

        cost = []
        open_list = [] # hold neighbors of the expanded nodes
        closed_list = [] # hold all the exapnded nodes

        habitats = habitat_list[:]
        habitat_open_list = habitat_list # hold haibitats that have not been explored 
        habitat_closed_list = [] # hold habitats that have been explored 

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
   
            if abs(current_node.pathLen - pathLenLimit) <= 10: # terminating condition
                path = []
                current = current_node

                while current is not None: # backtracking to find the d 
        
                    path.append(current.position)
                    cost.append(current.cost)
                    habitats_time_spent += self.habitats_time_spent(current)
                    current = current.parent
                    
                path_mps = [] 
        
                for point in path:
                    mps = Motion_plan_state(point[0], point[1])
                    path_mps.append(mps)
                
                trajectory = path_mps[::-1]
                
                print ("\n", "original trajectory: ", trajectory)
                print ("\n", "original trajectory length: ", len(trajectory))

                smoothPath = self.smoothPath(trajectory, habitats)

                return ([smoothPath, cost])
                # return (path_mps[::-1], cost)
            
            current_neighbors = self.curr_neighbors(current_node, boundary_list)

            children = []
            # grid = []

            '''Old Cost Function'''
            for neighbor in current_neighbors: # create new node if the neighbor is collision-free

                if self.collision_free(neighbor, obs_lst):

                    new_node = Node(current_node, neighbor)
                    result = self.update_habitat_coverage(new_node, habitat_open_list, habitat_closed_list)  # update habitat_open_list and habitat_closed_list
          
                    habitat_open_list = result[0]
                    habitat_closed_list = result[1]
                
                    cost_of_edge = cal_cost.cost_of_edge(new_node, habitat_open_list, habitat_closed_list, weights)
                    new_node.cost = new_node.parent.cost + cost_of_edge[0]
  
                    children.append(new_node)
           
            '''
            IMPLEMENTATION OF GRID MAP => DISCARDED FOR NOW
           
            for neighbor in current_neighbors: # create new node if the neighbor is collision-free
                
                if self.collision_free(neighbor, obs_lst):

                    """
                    SELECTION 
                    """ 
                    grid_val = self.create_grid_map(current_node, neighbor)   
                    grid.append((neighbor, grid_val))
                
            grid.remove(min(grid)) 
       

            append the selected neighbors to children 
            for neighbor in grid: 
                new_node = Node(current_node, neighbor[0])
                children.append(new_node)
                

                result = self.update_habitat_coverage(new_node, habitat_open_list, habitat_closed_list) # update habitat_open_list and habitat_closed_list
                habitat_open_list = result[0]
                habitat_closed_list = result[1]
            '''

            for child in children: 

                if child in closed_list:
                    continue

                result = cal_cost.cost_of_edge(child, habitat_open_list, habitat_closed_list, weights) 
                d_2 = result[1]
                d_3 = result[2]
                
                child.g = child.parent.cost - w2 * d_2 - w3 * d_3 
                child.cost = child.g
                child.h = - w2 * abs(pathLenLimit - child.pathLen) - w3 * len(habitat_open_list)
                child.f = child.g + child.h 
                child.pathLen = child.parent.pathLen + euclidean_dist(child.parent.position, child.position)
                child.time_stamp = int(child.pathLen/self.velocity)
                print ("\n", "child time stamp: ", child.time_stamp)

                # check if child exists in the open list and have bigger g 
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                x_pos, y_pos = self.get_indices(child.position[0], child.position[1])

                if self.visited_nodes[x_pos, y_pos] == 0: 
                    open_list.append(child)
                    self.visited_nodes[x_pos, y_pos] = 1

def main():
    weights = [0, 10, 10]
    start_cartesian = create_cartesian((33.445089, -118.486933), catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    #  convert to environment in casrtesian coordinates 
    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    
    obstacle_list = environ[0]
    boundary_list = environ[1]
    boat_list = environ[2]
    habitat_list = environ[3]

    astar_solver = astar(start, obstacle_list+boat_list, boundary_list, habitat_list) 
    final_path_mps = astar_solver.astar(habitat_list, obstacle_list+boat_list, boundary_list, start, 200, weights)

    print ("\n", "final trajectory: ",  final_path_mps[0])
    print ("\n", "Trajectory length: ", len(final_path_mps[0]))
    print ("\n", "cost of each node on the final trajectory: ",  final_path_mps[1])       

if __name__ == "__main__":
    main()