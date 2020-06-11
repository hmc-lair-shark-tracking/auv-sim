import math
import random
import timeit
from motion_plan_state import Motion_plan_state
import matplotlib.pyplot as plt
import numpy as np

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
        self.min_bound = boundary[0]
        self.max_bound = boundary[1]

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


    def curr_neighbors(self, current_node): 

        """
        Return a list of position tuples that are close to the current point

        Parameter:
            current_node: a Node object 
        """

        adjacent_squares = [(0,-10),(0,10),(-10,0),(10,0),(10,10),(10,-10),(-10,10),(-10,-10)]

        current_neighbors = []

        for new_position in adjacent_squares:
            
            node_position = (current_node.position[0]+new_position[0], current_node.position[1]+new_position[1])

            # check if it's within the boundary

            if node_position[0] >= self.min_bound.x and node_position[0] <= self.max_bound.x:
                if node_position[1] >= self.min_bound.y and node_position[1] <= self.max_bound.y:
                    current_neighbors.append(node_position)

        return current_neighbors  
    
    def path_distance(self, path):

        """
        Calculate the path distance 

        Parameter: 
            path - a list of Motion_plan_state objects, the trajectory created with A*
        """
        
        path_dist = 0 
        
        for index in range(len(path)-1):
           start_mps = path[index]
           end_mps = path[index+1]
           dist, _ = self.get_distance_angle(start_mps, end_mps)
           path_dist += dist

        return path_dist
           
    def perform_analysis(self, start, goal, boundary): 
        """
        Plot two graphs: the number of obstacles vs. A* unning time to find the optimal path
            and path distance vs. A* running time to find the optimal path 
        
        Parameter:
            start - a tuple of two elements: x and y coordinates
            goal - a tuple of two elements: x and y coordinates
            boundary - a list of two motion_plan_state objects 
        """

        obstacle_size = 1
        max_obstacle = 10
        obstacle_num = 0
        max_iteration = 30

        time_list = np.array([])
        ave_time_list = np.array([])
        obstacle_list = []
        obstacle_num_list = np.array([])

        path_dist_list = np.array([])
        ave_path_dist_list = np.array([])

        while obstacle_num < max_obstacle:

            obstacle_num += 1
            print ('obstacle_num: ', obstacle_num)
            obstacle_num_list = np.append(obstacle_num_list, obstacle_num)

            for iteration in range(max_iteration):

                i = 0 

                while i < obstacle_num:
                    obstacle_x = np.random.randint(0, 100, size=None, dtype='int')
                    obstacle_y = np.random.randint(0, 100, size=None, dtype='int')
                    if obstacle_x != goal[0] and obstacle_y != goal[1]:
                        obstacle_list.append(Motion_plan_state(obstacle_x, obstacle_y, size=obstacle_size))
                        i += 1
                        
                astar_solver = astar(start, goal, obstacle_list, boundary)
                start_time = timeit.timeit()
                path = astar_solver.astar(obstacle_list, start, goal)
                end_time = timeit.timeit()

                # compute and add path distance to list

                path_dist_list = np.append(path_dist_list, self.path_distance(path))
    
                time_list = np.append(time_list, abs(end_time - start_time))
            
            ave_time_list = np.append(ave_time_list, np.average(time_list))
            ave_path_dist_list = np.append(ave_path_dist_list, np.average(path_dist_list))
            print ('ave_time_list element: ', np.average(time_list))
            print ('ave_path_dist_list: ', np.average(path_dist_list))


        plot_obs = plt.figure(1)
        plt.plot(obstacle_num_list, ave_time_list) 
        plt.xlabel('obstacle number')
        plt.ylabel('running time')

        plot_dist = plt.figure(2)
        plt.plot(ave_path_dist_list, ave_time_list)
        plt.xlabel('path distance')
        plt.ylabel('running time') 

        plt.show()

    def astar(self, obs_lst, start, goal): 
        """
        Find the optimal path from start to goal avoiding given obstacles 

        Parameter: 
            obs_lst - a list of motion_plan_state objects that represent obstacles 
            start - a tuple of two elements: x and y coordinates
            goal - a tuple of two elements: x and y coordinates
        """

        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, goal)
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
   
            if current_node.position == end_node.position:
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
            
            current_neighbors = self.curr_neighbors(current_node)

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
                child.h = self.euclidean_dist(child.position, goal) # distance from child to end goal
                child.f = child.g + child.h

                # check if child exists in the open list and have bigger g 
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                open_list.append(child)
        
def main():

    start = (0,0)
    goal = (100,100)

    boundary = [Motion_plan_state(0,0), Motion_plan_state(100,100)]
    obstacle_list = [] 

    astar_solver = astar(start, goal, obstacle_list, boundary)

    astar_solver.perform_analysis(start, goal, boundary) 

    # obstacle_list = [Motion_plan_state(3,3,size=1),Motion_plan_state(3,6,size=2)]
    # astar_solver = astar(start, goal, obstacle_list, boundary)
    # final_path_mps = astar_solver.astar(obstacle_list, start, goal)
    # print (final_path_mps)

if __name__ == "__main__":
    main()
    
        



    
                




