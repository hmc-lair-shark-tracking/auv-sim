from shapely.geometry import Polygon
import math

from sharkOccupancyGrid import SharkOccupancyGrid, splitCell, plotShark
import catalina as catalina
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
    
    def update(self, curr, sharkGrid, traj_time_length, bin_interval, method):
        '''
        main function for shark occupancy grid estimate

        parameters:
            curr: current time bin
            sharkGrid: shark occupancy grid to be updated
            traj_time_length: Trajectory time length, i.e. the difference between the the last and first time stamp of the trajectory constructed
                i.e. how long the AUV will drive around for
            bin_interval: time interval to construct a separate shark occupancy grid, in seconds
            method: prediction method chosen to use to predict shark occupancy grid
                indicated by [method, prediction]
                method[0] = "ave" or "hist"
                method[1] = 1 or 2

        output: updated sharkGrid
        '''
        if method[0] == "ave":
            for _ in range(math.ceil(traj_time_length / bin_interval)):
                occGridPrev = sharkGrid[curr]
                occGridCurr = self.predictOnAve(occGridPrev, False, method[1], 0.6, 0.1)
                curr = (curr[0] + bin_interval, curr[1] + bin_interval)
                sharkGrid[curr] = occGridCurr
        
        elif method[0] == "hist":
            for _ in range(math.ceil(traj_time_length / bin_interval)):
                occGridPrev = sharkGrid[curr]
                occGridCurr = self.predictOnHist(occGridPrev, False, method[1], sharkGrid[(0, bin_interval)], 0.6, 0.1)
                curr = (curr[0] + bin_interval, curr[1] + bin_interval)
                sharkGrid[curr] = occGridCurr
        
        return sharkGrid

    def prediction1(self, occGridPrev, stayProb):
        '''
        Action update algorithm in Markov Localization, to predict position estimate 
            based on previous shark occupancy at xt-1, according to P(x'|t) = sum of P(x',t|j,t-1)P(j,t-1)
    
        parameters:
            occGridPrev: shark occupancy grid at time stamp t-1
            stayProb: probability the shark will stay in the cell at next time stamp
        '''
        #initialize the grid, probability initialized to 0
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]

        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            grid[row][col] = stayProb * occGridPrev[row][col]
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
                grid[row][col] += ((1 - stayProb) / len(neighbors)) * occGridPrev[neighbor[0]][neighbor[1]]

        return grid
    
    def prediction2(self, occGridPrev, k, gridInf):
        '''
        Action update algorithm in Markov Localization, to predict position estimate 
            based on previous shark occupancy at xt-1, according to P(x'|t) = P(x'|t-1) + k(P_inf - P(x'|t-1))
    
        parameters:
            occGridPrev: shark occupancy grid at time stamp t-1
            k: hyperparameter to be tuned
            gridInf: shark occupancy grid at infinite time stamp
        '''

        grid = occGridPrev.copy()

        for i in range(len(occGridPrev)):
            for j in range(len(occGridPrev[i])):
                grid[i][j] += k * (gridInf[i][j] - occGridPrev[i][j])
        
        return grid
    
    def predictOnAve(self, init_distribution, if_exp, method, stayProb, k):
        '''
        implementation of one method to predict future shark occupancy grid

        initial distribution for experiment: 1/n for each cell
        initial distribution for planner: shark occupancy grid at current time stamp

        prediction function1: random movement P(x'|t) = sum of P(x',t|j,t-1)P(j,t-1)
        prediction function2: proportional control P(x'|t) = P(x'|t-1) + k(P_inf - P(x'|t-1))where P_inf = 1/n for each cell

        parameters:
            init_distribution: initial distribution of shark occupancy grid, given if_exp parameter
            if_exp: boolean to indicate if it's for experiment or for planner
                True: init_distribution == None
                False: init_distribution == 1/n for each cell
            method: choose prediction function 1 or 2
                stayProb: parameter for method 1
                k: paramter for method2
        '''
        #initialize P_inf grid, probability initialized to 1/n
        minx, miny, maxx, maxy = self.boundary.bounds
        gridInf = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]
        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            gridInf[row][col] = 1 / len(self.cell_list)

        if if_exp:
            init_distribution = gridInf.copy()
        
        if method == 1:
            res = self.prediction1(init_distribution, stayProb)
        if method == 2:
            res = self.prediction2(init_distribution, k, gridInf)
        
        return [init_distribution, res]
    
    def predictOnHist(self, init_distribution, if_exp, method, gridInf, stayProb, k):
        '''
        implementation of one method to predict future shark occupancy grid

        initial distribution for experiment: shark occupancy grid from historical data
        initial distribution for planner: shark occupancy grid at current time stamp

        prediction function1: random movement P(x'|t) = sum of P(x',t|j,t-1)P(j,t-1)
        prediction function2: proportional control P(x'|t) = P(x'|t-1) + k(P_inf - P(x'|t-1))
            where P_inf = shark occupancy grid from historical data for each cell

        parameters:
            init_distribution: initial distribution of shark occupancy grid, given if_exp parameter
            if_exp: boolean to indicate if it's for experiment or for planner
                True: init_distribution == None
                False: init_distribution == shark occupancy grid from historical data for each cell
            method: choose prediction function 1 or 2
                stayProb: parameter for method 1
                k: paramter for method2
            gridInf: shark occupancy grid from historical data for each cell
        '''
        if if_exp:
            init_distribution = gridInf.copy()
        
        if method == 1:
            res = self.prediction1(init_distribution, stayProb)
        if method == 2:
            res = self.prediction2(init_distribution, k, gridInf)
        
        return [init_distribution, res]

    
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

# boundary_poly = []
# for b in catalina.BOUNDARIES:
#     pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
#     boundary_poly.append((pos[0],pos[1]))
# boundary_poly = Polygon(boundary_poly)
# shark_dict2 = {1: [Motion_plan_state(-120 + (0.1 * i), -60 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-90 - (0.1 * (i - 301)), -30 + (0.15 * (i - 301)), traj_time_stamp=i) for i in range(302,501)], 
#     2: [Motion_plan_state(-65 - (0.1 * i), -50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-95 + (0.15 * (i - 301)), -20 + (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)],
#     3: [Motion_plan_state(-110 + (0.1 * i), -40 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-80 + (0.15 * (i - 301)), -70 + (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)], 
#     4: [Motion_plan_state(-105 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-135 + (0.12 * (i - 301)), -25 + (0.07 * (i - 301)), traj_time_stamp=i) for i in range(302,501)],
#     5: [Motion_plan_state(-120 + (0.1 * i), -50 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-90 + (0.11 * (i - 301)), -80 + (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)], 
#     6: [Motion_plan_state(-85 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-115 - (0.09 * (i - 301)), -25 - (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)],
#     7: [Motion_plan_state(-270 + (0.1 * i), 50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-240 - (0.08 * (i - 301)), 80 + (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)], 
#     8: [Motion_plan_state(-250 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-280 - (0.1 * (i - 301)), 105 - (0.1 * (i - 301)), traj_time_stamp=i) for i in range(302,501)],
#     9: [Motion_plan_state(-260 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-290 + (0.08 * (i - 301)), 105 + (0.07 * (i - 301)), traj_time_stamp=i) for i in range(302,501)], 
#     10: [Motion_plan_state(-275 + (0.1 * i), 80 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-245 - (0.13 * (i - 301)), 50 - (0.12 * (i - 301)), traj_time_stamp=i) for i in range(302,501)]}
# testing = SharkOccupancyGrid(shark_dict2, 10, boundary_poly, 50, 50)
# occGrid = testing.constructSharkOccupancyGrid(shark_dict2[1])
# updating = SharkUpdate(boundary_poly, 10, testing.cell_list)
# res = updating.predictOnAve(None, True, 1, 0.6, 0.1)
# res = updating.predictOnHist(None, True, 2, occGrid, 0.6, 0.1)
# testing.plot({"current": res[0], "future": res[1]})