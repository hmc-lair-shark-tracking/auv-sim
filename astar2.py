import math
from motion_plan_state import Motion_plan_state
import matplotlib.pyplot as plt

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
        dx = abs(point1[0]-point2[0])
        dy = abs(point1[1]-point2[1])

        return math.sqrt(dx*dx+dy*dy)


    def get_distance_angle(self, start_mps, end_mps):
        dx = end_mps.x-start_mps.x
        dy = end_mps.y-start_mps.y
        
        dist = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy,dx)
        return dist, theta

    def check_collision(self, position, obstacleList): # obstacleList lists of motion plan state 

        position_mps = Motion_plan_state(position[0], position[1])

        for obstacle in obstacleList:
            d, _ = self.get_distance_angle(obstacle, position_mps)
            if d <= obstacle.size:
                return False # collision 

        return True

    def curr_neighbors(self, current_node): 

        """Return a list of points that are close to the current point"""

        adjacent_squares = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]

        current_neighbors = []

        for new_position in adjacent_squares:
            
            node_position = (current_node.position[0]+new_position[0], current_node.position[1]+new_position[1])

            # check if it's within the boundary

            if node_position[0] >= self.min_bound.x and node_position[0] <= self.max_bound.x:
                if node_position[1] >= self.min_bound.y and node_position[1] <= self.max_bound.y:
                    continue

            current_neighbors.append(node_position)

        return current_neighbors

    def astar(self, obs_lst, start, goal): # obs_lst is a list of motion plan state; start, goal are position tuples

        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, goal)
        end_node.g = end_node.h = end_node.f = 0

        open_list = [] # hold neighbors of the expanded nodes
        closed_list = [] # hold all the exapnded nodes 

        open_list.append(start_node)

        # print(open_list[0].position)

        while len(open_list) > 0:
            
            current_node = open_list[0] # initialize the current node
            current_index = 0

            for index, item in enumerate(open_list): # find the current node with the smallest f
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            
            open_list.pop(current_index)
            # print([node.position for node in open_list])

            closed_list.append(current_node)
            # print([node.position for node in closed_list]) 

            if current_node == end_node:
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
   
            # print ([child.position for child in children])

            for child in children: 
                for closed_child in closed_list:
                    if child == closed_child: 
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
    goal = (7,6)

    boundary = [Motion_plan_state(0,0), Motion_plan_state(10,10)]

    obstacle_list = [Motion_plan_state(5,5,size=1),Motion_plan_state(3,6,size=2)]
    
    astar_solver = astar(start, goal, obstacle_list, boundary)

    final_path_mps = astar_solver.astar(obstacle_list, start, goal)

    print (final_path_mps)

if __name__ == "__main__":
    main()
    
        



    
                




