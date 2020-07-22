from shapely.geometry import Polygon
import math

from path_planning.sharkOccupancyGrid import SharkOccupancyGrid, splitCell
import path_planning.catalina as catalina
from motion_plan_state import Motion_plan_state

class SharkUpdate:
    '''
        a class to produce shark occupancy value at time stamp t, from now to future
            based on Markov Localization and particle filter
    '''
    def __init__(self, boundary, cell_size, cell_list):
        '''
        paramters:
            boundary: a Polygon object representing the work space, needed to be splitted into cells
            cell_size: cell size in meters
        '''
        self.cell_size = cell_size
        self.boundary = boundary
        self.cell_list = cell_list
    
    def prediction(self, occGridPrev):
        '''
        Action update algorithm in Markov Loclization, to predict position estimate 
            based on previous shark occupancy at xt-1 and odometry measurement

        odometryMeasurement: the shark has 90% chance to stay in the same cell at time stamp t-1
            and an equal chance to move in any one direction: East/West/North/South

        parameters:
            occGridPrev: shark occupancy grid at time stamp t-1
        '''
        #initialize the grid, probability initialized to 0.01
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]

        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            grid[row][col] = 0.6 * occGridPrev[row][col]
            neighbors = []
            if (row - 1) >= 0 and grid[row-1][col] != 0:
                neighbors.append((row-1, col))
            if (col - 1) >= 0 and grid[row][col-1] != 0:
                neighbors.append((row, col-1))
            if (row + 1) < len(grid) and grid[row+1][col] != 0:
                neighbors.append((row+1, col))
            if (col + 1) < len(grid[0]) and grid[row][col+1] != 0:
                neighbors.append((row, col+1))
            for neighbor in neighbors:
                grid[row][col] += (0.4 / len(neighbors)) * occGridPrev[neighbor[0]][neighbor[1]]

        return grid
    
    def correction(self, particles, prediction):
        '''
        perception update algorithm using Bayes Filter, to correct position estimate x at time stamp t using exteroceptive sensor/range measurement

        rangeMeasurement: using particle filter to get the number of particles at a given cell

        parameters:
            prediction: shark position estimate x_t at time stamp t 
            particles: the number of particles in each cell 
        '''
        #initialize the grid, probability initialized to 0.01
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]
        tempSum = 0

        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            #likelihood P(z|x) = number of particles in a cell / total number of particles
            likelihood = particles[row][col] / 1000
            #prior P(x')
            prior = prediction[row][col]
            #P(x|z) = P(z|x) * P(x')
            grid[row][col] = likelihood * prior
            tempSum += grid[row][col]
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j] = grid[i][j] / tempSum

        return grid

    def cellToIndex(self, cell):
        minx, miny, _, _ = self.boundary.bounds
        lowx, lowy, _, _ = cell.bounds
        col = int((lowx - minx) / self.cell_size)
        row = int((lowy - miny) / self.cell_size)
        return (row, col)

boundary_poly = []
for b in catalina.BOUNDARIES:
    pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
    boundary_poly.append((pos[0],pos[1]))
boundary_poly = Polygon(boundary_poly)
shark_dict = {1: [Motion_plan_state(-120 + (0.2 * i), -60 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    2: [Motion_plan_state(-65 - (0.2 * i), -50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    3: [Motion_plan_state(-110 + (0.2 * i), -40 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    4: [Motion_plan_state(-105 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    5: [Motion_plan_state(-120 + (0.2 * i), -50 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    6: [Motion_plan_state(-85 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    7: [Motion_plan_state(-270 + (0.2 * i), 50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    8: [Motion_plan_state(-250 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    9: [Motion_plan_state(-260 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    10: [Motion_plan_state(-275 + (0.2 * i), 80 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)]}
testing = SharkOccupancyGrid(shark_dict, 10, boundary_poly, 50, 50)
print('finish')
occGrid = testing.constructSharkOccupancyGrid(shark_dict[5])
print('finish')
updating = SharkUpdate(boundary_poly, 10, testing.cell_list)
print('finish')
occGridCurr = updating.prediction(occGrid)
print('finish')
testing.plot({"prev": occGrid, "curr": occGridCurr})