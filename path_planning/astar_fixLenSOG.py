import math
import geopy.distance
import random
import timeit
import csv 
import matplotlib.pyplot as plt
import numpy as np
import catalina

from motion_plan_state import Motion_plan_state
from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon 
from catalina import create_cartesian
from sharkOccupancyGrid import SharkOccupancyGrid, splitCell
from matplotlib import cm, patches, collections

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

def createSharkGrid(filepath, cell_list):
    test = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            string = row["time bin"]
            strings = string.split(", ")
            key = []
            key.append(int(strings[0][1:]))
            key.append(int(strings[1][0:len(strings[1])-1]))

            temp = row["grid"][1:len(row["grid"])-1]
            temp = temp.split(", ")
            val = {}
            for i in range(len(temp)-1): # initially was len(temp); changed to fix outofindex error
                val[cell_list[i].bounds] = float(temp[i])
            test[(key[0],key[1])] = val
    return test

def plot(sharkOccupancyGrid, grid_dict, A_star_traj):

    fig = plt.figure(1, figsize=(10,15))
    x,y = sharkOccupancyGrid.boundary.exterior.xy

    astar_x_array = []
    astar_y_array = []

    for point in A_star_traj:
        astar_x_array.append(round(point.x, 2))
        astar_y_array.append(round(point.y, 2))

    for i in range(len(list(grid_dict.keys()))):
        ax = fig.add_subplot(5, 2, i+1)
        ax.plot(x, y, color="black")
        plt.plot(astar_x_array, astar_y_array, marker = ',', color = 'r')

        patch = []
        occ = []
        key = list(grid_dict.keys())[i]
        for cell in sharkOccupancyGrid.cell_list:
            polygon = patches.Polygon(list(cell.exterior.coords), True)
            patch.append(polygon)
            row, col = sharkOccupancyGrid.cellToIndex(cell)
            occ.append(grid_dict[key][row][col])

        p = collections.PatchCollection(patch)
        p.set_cmap("Greys")
        p.set_array(np.array(occ))
        ax.add_collection(p)
        fig.colorbar(p, ax=ax)

        ax.set_xlim([sharkOccupancyGrid.boundary.bounds[0]-10, sharkOccupancyGrid.boundary.bounds[2]+10])
        ax.set_ylim([sharkOccupancyGrid.boundary.bounds[1]-10, sharkOccupancyGrid.boundary.bounds[3]+10])

        ax.title.set_text(str(list(grid_dict.keys())[i]))
    
        for shark_id, traj in sharkOccupancyGrid.timeBinDict[key].items():
            ax.plot([mps.x for mps in traj], [mps.y for mps in traj], label=shark_id)

    plt.legend(loc="lower right")
    plt.show()

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
        self.time_stamp = 0 # time stamp stores the amount of time passed by at each Node object 

class singleAUV:

    def __init__(self, start, habitatList, obstacleList, boundaryList, sharkGrid, AUV_velocity):
        
        self.start = start
        self.velocity = AUV_velocity
        self.obstacle_list = obstacleList
        self.boundary_list = boundaryList
        self.habitat_open_list = habitatList.copy()
        self.habitat_closed_list = []
        self.visited_nodes = np.zeros([600, 600])

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
        
    def euclidean_dist(self, point1, point2): 
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
        # return False if not within boundary 
        return False 
        
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
                # return False if collision 
                return False 

        return True 

    def Walkable(self, check_point, current_point, obstacleList):
        """
        Return true if there's no obstacle from check point to current point => walkable;
        return false otherwise

        Parameter:
            check_point: a Motion_plan_state object
            current_point: a Motion_plan_state object
        """
        start_x = check_point.x
        start_y = check_point.y

        step_x = int(abs(current_point.x - check_point.x)/5)
        step_y = int(abs(current_point.y - check_point.y)/5)
    
        while start_x <= current_point.x and start_y <= current_point.y:
            intermediate = (start_x, start_y)
          
            start_x += step_x
            start_y += step_y

            if not self.collision_free(intermediate, obstacleList): 
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

            # check if within the boundary
            if self.within_bounds(boundary_list, node_position):
                current_neighbors.append(node_position)

        return current_neighbors  

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
            if dist <= item.size: 
                habitat_open_list.pop(index)
                habitat_closed_list.append(item)

        return (habitat_open_list, habitat_closed_list)
    
    def inside_habitats(self, mps, habitats):
        """
        helper function to smooth the path;
        return True if the location mps is inside any of the habitats in habitat_list

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

            if self.Walkable(checkPoint, currentPoint, self.obstacle_list): # if no obstacle in between two points
                
                inside_habitats = self.inside_habitats(currentPoint, haibitats)
           
                if not inside_habitats: # if currentPoint is NOT within any of the habitats => removable
                    temp = currentPoint
                    index += 1 
                    currentPoint = trajectory[index]                    
                    smoothTraj.remove(temp)
 
                else: 
                    index += 1
                    currentPoint = trajectory[index]
                    
            else:
                checkPoint = currentPoint
                index += 1
                currentPoint = trajectory[index]

        return smoothTraj
    
    def findCurrSOG(self, grid, curr_time_stamp):
        """
        Return the SOG that corresponds to the curr_time_stamp

        Parameter: 
            grid: a dictionary of shark occupancy grid maps
                key: timebin, tuple(start time, end time)
                value: a dictionary representing occupancy grid of each shark during this time bin
            curr_time_stamp: an integer
        """
        
        for time, AUVGrid in self.sharkGrid.items(): # time: tuple(start time, end time); value: a dictionary representing occupancy grid of each shark during this time bin
            
            # print ("\n", "time: ", time, "current time stamp: ", curr_time_stamp)

            if self.with_in_time_bin(time, curr_time_stamp):
                
                return AUVGrid

    def findCurrAUVGrid(self, grid_dict, child_node):

        curr_time_stamp = child_node.time_stamp

        for key_time, AUVGrid in grid_dict.items():
            if self.with_in_time_bin(key_time, curr_time_stamp):
                return AUVGrid

    def get_max_prob_from_grid(self, curr_grid):
        """
        Return the maximum probability from the given grid

        Parameter:
            curr_grid: a dictionary representing occupancy grid of each shark during this time bin
        """
        max_value = 0

        for grid_val in curr_grid.values():
            if grid_val > max_value:
                max_value = grid_val

        # print ("\n", "max probability: ", max_value)
        return max_value

    def get_cell_prob(self, node, currGrid):
        """
        Return the probability of the cell corresponding to the node in AUV detection grid/currGrid

        Parameter: 
            node: a Node object
            currGrid: a dictionary representing occupancy grid of each shark during this time bin
        """
        x_test = node.position[0]
        y_test = node.position[1]

        key = None
       
        for pos_tuple in list(currGrid.keys()):
      
            pos_1 = (round(pos_tuple[0], 2), round(pos_tuple[1], 2)) # pos_1 = (x1, y1)
            pos_2 = (round(pos_tuple[2], 2), round(pos_tuple[3], 2)) # pos_2 = (x2, y2)
          
            dx = abs(pos_1[0] - pos_2[0])
            dy = abs(pos_1[1] - pos_2[1])
 
            if (abs(x_test - pos_1[0]) <= dx and abs(x_test - pos_2[0]) <= dx) and (abs(y_test - pos_1[1]) <= dy and abs(y_test - pos_2[1]) <= dy):
                    key = pos_tuple
                    break

        if key == None:
            print ("ERROR! Cannot find corresponding location")
        else:
            # print ("\n", "cell probability: ", currGrid[key])
            return currGrid[key]
     
    def get_top_n_prob(self, n, currGrid):
        """
        Return the sum of top n probabilities from the currGrid

        Parameter: 
            n: an integer; indicate the number of steps left 
            currGrid: a dictionary representing occupancy grid of each shark during this time bin
        """

        total = 0
        probabilities = list(currGrid.values())
        probabilities.sort(reverse=True)
        
        for index in range(n):
            total += probabilities[index]

        # print ("\n", "top n probabilities: ", total)
        return total

    def with_in_time_bin(self, time_bin, curr_time_stamp):
        """
        Return True if current_time_stamp is within the range given by time_bin; 
        return False otherwise

        Parameter:
            time_bin: a tuple of two integers; time_bin=(start_time, end_time)
            curr_time_stamp: an integer
        """
        start_time = time_bin[0]
        end_time = time_bin[1]
        if curr_time_stamp <= end_time and curr_time_stamp >= start_time:
            return True
        else:
            return False
            
    def astar(self, pathLenLimit, weights): 
        """
        Find the optimal path from start to goal avoiding given obstacles 
        Parameter: 
            pathLenLimit: in meters; the length limit of the A* trajectory 
            weights: a list of three numbers [w1, w2, w3] 
            shark_traj: a list of Motion_plan_state objects 
        """
        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]
        w4 = weights[3]

        start_node = Node(None, self.start)
        start_node.g = start_node.h = start_node.f = 0

        cost = []
        open_list = [] # hold neighbors of the expanded nodes
        closed_list = [] # hold all the exapnded nodes

        # habitats = self.habitat_list[:]
        # habitat_open_list = self.habitat_list[:] # hold haibitats that have not been explored 
        # habitat_closed_list = [] # hold habitats that have been explored 

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
        
                    path.append(current)
                    # update habitat_open_list 
                    habitats_open_n_close_list = self.update_habitat_coverage(current_node, self.habitat_open_list, self.habitat_closed_list)
                    self.habitat_open_list = habitats_open_n_close_list[0]
                    self.habitat_closed_list = habitats_open_n_close_list[1]
                    cost.append(current.cost)
                    
                    current = current.parent
                    
                path_mps = [] 
        
                for node in path:
                    mps = Motion_plan_state(node.position[0], node.position[1], traj_time_stamp=round(node.time_stamp, 2))
                    path_mps.append(mps)
                
                trajectory = path_mps[::-1]
                
                # print ("\n", "Original Trajectory: ", trajectory)
                # print ("\n", "Original Trajectory length: ", len(trajectory))

                smoothPath = self.smoothPath(trajectory, self.habitat_open_list)
                
                # return {"path length" : len(trajectory), "path" : trajectory, "cost" : cost[0], "cost list" : cost}
                return {"path length" : len(smoothPath), "path" : trajectory, "cost" : cost[0], "cost list" : cost, "node" : path[::-1]} 
                
            current_neighbors = self.curr_neighbors(current_node, self.boundary_list)

            children = []

            for neighbor in current_neighbors: # create new node if the neighbor is collision-free

                if self.collision_free(neighbor, self.obstacle_list):

                    new_node = Node(current_node, neighbor)

                    children.append(new_node)
           
            for child in children: 

                if child in closed_list:
                    continue
                
                child.pathLen = child.parent.pathLen + euclidean_dist(child.parent.position, child.position)
                dist_left = abs(pathLenLimit - child.pathLen)
                child.time_stamp = int(child.pathLen/self.velocity)
                currGrid = self.findCurrSOG(self.sharkGrid, child.time_stamp) 
                
                child.g = child.parent.cost - w4 * self.get_cell_prob(child, currGrid)
                child.cost = child.g
                # child.h = - w2 * dist_left - w3 * len(habitat_open_list) - w4 * dist_left * self.get_max_prob_from_grid(currGrid)
                child.h = - w2 * dist_left - w3 * len(self.habitat_open_list) - w4 * self.get_top_n_prob(int(dist_left), currGrid)
                child.f = child.g + child.h 

                # check if child exists in the open list and have bigger g 
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                
                x_pos, y_pos = self.get_indices(child.position[0], child.position[1])

                if self.visited_nodes[x_pos, y_pos] == 0: 
                    open_list.append(child)
                    self.visited_nodes[x_pos, y_pos] = 1

def main():
    weights = [0, 10, 10, 100]
    start_cartesian = create_cartesian((33.446198, -118.486652), catalina.ORIGIN_BOUND)
    start = (round(start_cartesian[0], 2), round(start_cartesian[1], 2))
    print ("start: ", start) 

    #  convert to environment in casrtesian coordinates 
    environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)
    
    obstacle_list = environ[0]
    boundary_list = environ[1]
    boat_list = environ[2]
    habitat_list = environ[3]

   # testing data for shark trajectories
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

    astar_solver = singleAUV(start, habitat_list, obstacle_list+boat_list, boundary_list, {}, AUV_velocity=1) 
    final_path_mps = astar_solver.astar(50, weights)

    print ("\n", "Open Habitats: ", astar_solver.habitat_open_list)
    print ("\n", "Closed Habitats: ", astar_solver.habitat_closed_list)
    print ("\n", "Final Trajectory: ",  final_path_mps["path"])
    print ("\n", "Trajectory Length: ", final_path_mps["path length"])
    print ("\n", "Trajectory Cost: ",  final_path_mps["cost"])
    print ("\n", "Trajectory Cost List: ", final_path_mps["cost list"])

    boundary_poly = []
    for pos in boundary_list:
        boundary_poly.append((pos.x, pos.y))

    boundary = Polygon(boundary_poly) # a Polygon object that represents the boundary of our workspace 
    sharkOccupancyGrid = SharkOccupancyGrid(shark_dict, 10, boundary, 50, 50)

    grid_dict = sharkOccupancyGrid.convert()
    plot(sharkOccupancyGrid, grid_dict[0], final_path_mps["path"])

if __name__ == "__main__":
    main()