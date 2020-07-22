import math
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point, LinearRing, LineString, MultiPolygon
from shapely.ops import split
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib import cm, patches, collections
import numpy as np
import csv
import time

import catalina
from motion_plan_state import Motion_plan_state

class SharkOccupancyGrid:
    '''
    a class to construct shark occupancy grid maps in each time bin to represent the distribution of shark positions 
    
    the map will be pre-constructed before planning starts using shark trajectories
    each time cost function involving keeping track of sharks in hydrophone range is calculated, corresponding occupancy 
        grid will be used
    '''

    def __init__(self, shark_dict, cell_size, boundary, bin_interval, detect_range):
        '''
        create a occupancy dictionary to store occupancy grid maps at different time bins
        key is time bin, value is the corresponding occupancy grid map

        paramaters:
            shark_dict: a dictionary representing shark trajectories, key is shark_id and value is the corresponding trajectory
            cell_size: cell size in meters
            boundary: a Polygon object representing the configuration space, needed to be splitted into cells
            bin_interval: time interval to construct a separate shark occupancy grid, in seconds
            detect_range: the maximum distance hydrophone can track sharks, in meters
        the probability of occupancy of each cell is calculated by the time sharks spent in the cell / duration of time bin
        '''
        self.data = shark_dict
        self.cell_size = cell_size
        self.cell_list = splitCell(boundary, cell_size)
        self.bin_interval = bin_interval
        self.detect_range = detect_range
        self.boundary = boundary

        self.bin_list = self.createBinList()
    
    def convert(self):
        '''
        convert a dictionary of shark trajectories
            key: shark ID, int
            value: shark trajectory, a list of Motion_plan_state
        
        output: 
        sharkOccupancy:
            a dictionary of shark occupancy grid maps
                key: timebin, tuple(start time, end time)
                value: a dictionary representing occupancy grid of each shark during this time bin
        '''

        #convert to a dictionary, whose key is time bin
        self.timeBinDict = self.convertToTimeBin()

        resultArr = {}
        resultCell = {}
        for time, traj_dict in self.timeBinDict.items():
            grid = self.constructGrid(traj_dict)
            # grid = self.simplifyGrid(grid)
            resultArr[time] = grid
            resultCell[time] = self.convert2DArr(grid)
        return (resultArr, resultCell)

    def splitCell(self, geometry, count=0):
        """Split a Polygon into two parts across it's shortest dimension
        
        parameter:
            geometry: the polygon of configuration space to be splitted into cells
            
        output: a list of Polygon objects, represented cells in the configuration space
        """
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        if max(width, height) <= self.cell_size or count == 250:
            # either the polygon is smaller than the cell size, or the maximum
            # number of recursions has been reached
            return [geometry]

        if height >= width:
            # split left to right
            a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
            b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
        else:
            # split top to bottom
            a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
            b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])

        result = []
        for d in (a, b,):
            c = geometry.intersection(d)
            if not isinstance(c, GeometryCollection):
                c = [c]
            for e in c:
                if isinstance(e, (Polygon, MultiPolygon)):
                    result.extend(self.splitCell(e, count+1))
        if count > 0:
            return result

        # convert multipart into singlepart
        final_result = []
        for g in result:
            if isinstance(g, MultiPolygon):
                final_result.extend(g)
            else:
                final_result.append(g)
        return final_result
    
    def constructGrid(self, shark_traj_dict):
        '''
        create a single AUV detecting grid at during one time bin for multiple sharks
        the detecting probability of each cell represents the probability AUV can detect sharks in this cell
            calculated by multiply probability of this cell for each cell together
        
        parameters:
            shark_traj_dict: a dictionary stored separate shark trajectories within the single time bin
        output: a dictionary representing the AUV detecting probability of each cell
            key: Polygon object representing each cell
            value: occupancy probability of the cell, calculated by num of sharks * average time sharks spent in this cell
        '''
        #initialize AUV detecing grid
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]

        for shark_id, traj in shark_traj_dict.items():
            tempOccGrid = self.constructSharkOccupancyGrid(traj)
            tempAUVGrid = self.constructAUVGrid(tempOccGrid)
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    grid[i][j] = grid[i][j] + tempAUVGrid[i][j]

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                grid[i][j] = grid[i][j] / len(list(shark_traj_dict.keys()))
        
        return grid
    
    def constructAUVGrid(self, occGrid):
        '''
        construct a AUV detecting probability grid for a single shark during one time bin
            AUV detecting probability represents the probability that if AUV resides in this cell,
            the probability it can detect this shark
            calculated by the sum of shark occupancy of all nearby cells within hydrophone range
        
        for each cell in the work space, sum up the occupancy of cells within detect_range if AUV resides in this cell

        parameter:
            occGrid: occupancy grid for a single shark during one time bin

        output: an AUV detecting grid representing AUV detecting probability of each cell
            note: AUV detecting probability of each cell should be no larger than 1
        '''
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]
        count = int(math.ceil(self.detect_range / self.cell_size))
        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            row_min, col_min = row - (2*count), col - (2*count)
            for i in range(2 * (2*count)):
                row_temp = row_min + i
                for j in range(2 * 2 * count):
                    col_temp = col_min + j
                    if row_temp >=0 and row_temp < len(grid) and col_temp >= 0 and col_temp < len(grid[0]):
                        if math.sqrt((row_temp - row)**2 + (col_temp - col)**2) <= count:
                            grid[row][col] += occGrid[row_temp][col_temp]
            if grid[row][col] > 1:
                print("problem!", grid[row][col])
        return grid
    
    def constructSharkOccupancyGrid(self, traj):
        '''
        construct occupancy grid for a single shark
            occupancy represents the probability one shark resides in this cell during the time bin
            occupancy of all cells should sum up to 1, assuming the shark is always in the boundary of the work space
        
        parameter:
            traj: single shark trajectory, a list of motion_plan_state

        output:
            an occupancy grid representing the occupancy of this shark at each cell during the time bin
        '''
        #initialize the grid, probability initialized to 0.01
        minx, miny, maxx, maxy = self.boundary.bounds
        grid = [[0 for _ in range(int(math.ceil(maxx - minx) / self.cell_size)+1)] for _ in range(int(math.ceil(maxy - miny) / self.cell_size)+1)]
        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            grid[row][col] = 0.01
        
        #normalize factor
        nor = (len(traj) + len(self.cell_list) * 0.01)

        #calculate occupancy probability
        for point in traj:
            for cell in self.cell_list:
                point = Point(point.x, point.y)
                if point.within(cell) or cell.touches(point):
                    row, col = self.cellToIndex(cell)
                    grid[row][col] += 1
                    break
        
        temp_sum = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j] = grid[i][j] / nor
                temp_sum += grid[i][j]
        # print(temp_sum)
        return grid

    def splitTraj(self, traj):
        '''
        split trajectory based on time bin
        '''
        #initialize 
        new_traj = [[self.bin_list[i]] for i in range(len(self.bin_list))]

        for point in traj:
            time = point.traj_time_stamp
            for i in range(len(self.bin_list)):
                if time >= self.bin_list[i][0] and time <=  self.bin_list[i][1]:
                    new_traj[i].append(point)
                    break
        return new_traj

    def convertToTimeBin(self):
        '''
        convert a dictionary of shark trajectories
            key: shark ID, int
            value: shark trajectory, a list of Motion_plan_state
        
        output: a dictionary of shark trajectories
            key: timebin, tuple(start time, end time)
            value: a dictionary representing shark trajectories during this timebin
                key: shark ID
                value: shark trajectory of corresponding shark in this time bin
        '''
        #initialize a dictionary
        result = {}
        for item in self.bin_list:
            result[item] = {}
        
        for shark, traj in self.data.items():
            new_traj_list = self.splitTraj(traj)
            for new_traj in new_traj_list:
                result[new_traj[0]][shark] = new_traj[1:]

        return result
    
    def convert2DArr(self, arr):
        result = {}
        for cell in self.cell_list:
            row, col = self.cellToIndex(cell)
            if arr[row][col] == 0:
                continue
            else:
                result[cell.bounds] = arr[row][col]
        return result
    
    def cellToIndex(self, cell):
        minx, miny, _, _ = self.boundary.bounds
        lowx, lowy, _, _ = cell.bounds
        col = int((lowx - minx) / self.cell_size)
        row = int((lowy - miny) / self.cell_size)
        return (row, col)

    def indexToCell(self, row, col):
        minx, miny, _, _ = self.boundary.bounds
        lowx = (col * self.cell_size) + minx
        lowy = (row * self.cell_size) + miny
        return (lowx, lowy, lowx+self.cell_size, lowy+self.cell_size)
    
    def createBinList(self):
        '''
        create a list of tuples representing each time bin
        '''
        longest_time = 0
        for _, traj in self.data.items():
            if traj[-1].traj_time_stamp > longest_time:
                longest_time = traj[-1].traj_time_stamp

        n_expand = math.floor(longest_time / self.bin_interval)
        bin_list = []
        for i in range(n_expand):
            bin_list.append((i * self.bin_interval, (i + 1) * self.bin_interval))
        
        return bin_list
    
    def simplifyGrid(self, grid):
        '''
        simplify the grid dictionary such that only occupied cells will show up
        '''
        for cell in list(grid):
            if grid[cell] == 0:
                del grid[cell]
        
        return grid

    def plot(self, grid_dict):
        fig = plt.figure(1, figsize=(10, 10))
        x,y = self.boundary.exterior.xy
        for i in range(len(list(grid_dict.keys()))):
            ax = fig.add_subplot(4, 4, i+1)
            ax.plot(x, y, color="black")

            patch = []
            occ = []
            key = list(grid_dict.keys())[i]
            for cell in self.cell_list:
                polygon = patches.Polygon(list(cell.exterior.coords), True)
                patch.append(polygon)
                row, col = self.cellToIndex(cell)
                occ.append(grid_dict[key][row][col])

            p = collections.PatchCollection(patch)
            p.set_cmap("Greys")
            p.set_array(np.array(occ))
            ax.add_collection(p)
            fig.colorbar(p, ax=ax)

            ax.set_xlim([self.boundary.bounds[0]-10, self.boundary.bounds[2]+10])
            ax.set_ylim([self.boundary.bounds[1]-10, self.boundary.bounds[3]+10])

            ax.title.set_text(str(list(grid_dict.keys())[i]))
        
            for shark_id, traj in self.timeBinDict[key].items():
                ax.plot([mps.x for mps in traj], [mps.y for mps in traj], label=shark_id)
        
        plt.legend(loc="lower right")
        plt.show()
    
def plotShark(boundary, shark_dict):
    plt.figure()
    x,y = boundary.exterior.xy
    plt.plot(x, y, color="black")
    for sharkID, traj in shark_dict.items():
        plt.plot([mps.x for mps in traj], [mps.y for mps in traj], label=sharkID)
    
    plt.legend()
    plt.show()

def splitCell(polygon, cell_size):
    minx, miny, maxx, maxy = polygon.bounds
    horizontal_splitters = []
    while (miny + cell_size) <= maxy:
        horizontal_splitters.append(LineString([(minx, miny + cell_size), (maxx, miny + cell_size)]))
        miny = miny + cell_size
    vertical_splitters = []
    minx, miny, maxx, maxy = polygon.bounds
    while (minx + cell_size) <= maxx:
        vertical_splitters.append(LineString([(minx + cell_size, miny), (minx + cell_size, maxy)]))
        minx += cell_size

    splitters = horizontal_splitters + vertical_splitters
    result = polygon
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
        
    return result

def getGridByTime(time, gridDict):
    '''
    Given a specific time stamp, find the corresponding shark occupancy grid / AUV detecting grid in the dictionary

    parameters:
        time: time stamp
        gridDict: a dictionary, whose key is time bin and value is the corresponding grid during this time bin
    '''
    for timebin, grid in gridDict.items():
        if time >= timebin[0] and time <= timebin[1]:
            return grid

def getGridByInterval(interval, gridDict):
    '''
    Given a time interval, find all shark occupancy grids / AUV detecting grids within this time interval

    parameters:
        interval: time interval
        gridDict: a dictionary, whose key is time bin and value is the corresponding grid during this time bin
    
    output:
        res: a distionary consisting of all grids in this interval
    '''
    res = {}    
    for time_bin in gridDict:
        if (interval[0] >= time_bin[0] and interval[0] <= time_bin[1]) or (time_bin[0] >= interval[0] and time_bin[1] <= interval[1]) or(interval[1] >= time_bin[0] and interval[1] <= time_bin[1]):
            res[time_bin] = gridDict[time_bin]
    
    return res

# boundary_poly = []
# for b in catalina.BOUNDARIES:
#     pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
#     boundary_poly.append((pos[0],pos[1]))
# boundary_poly = Polygon(boundary_poly)
# shark_dict1 = {1: [Motion_plan_state(-120 + (0.2 * i), -60 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
#     2: [Motion_plan_state(-65 - (0.2 * i), -50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
#     3: [Motion_plan_state(-110 + (0.2 * i), -40 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
#     4: [Motion_plan_state(-105 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
#     5: [Motion_plan_state(-120 + (0.2 * i), -50 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
#     6: [Motion_plan_state(-85 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
#     7: [Motion_plan_state(-270 + (0.2 * i), 50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
#     8: [Motion_plan_state(-250 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
#     9: [Motion_plan_state(-260 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
#     10: [Motion_plan_state(-275 + (0.2 * i), 80 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)]}
# shark_dict2 = {1: [Motion_plan_state(-120 + (0.1 * i), -60 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-90 - (0.1 * i), -30 + (0.15 * i), traj_time_stamp=i) for i in range(302,501)], 
#     2: [Motion_plan_state(-65 - (0.1 * i), -50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-95 + (0.15 * i), -20 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
#     3: [Motion_plan_state(-110 + (0.1 * i), -40 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-80 + (0.15 * i), -70 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
#     4: [Motion_plan_state(-105 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-135 + (0.12 * i), -25 + (0.07 * i), traj_time_stamp=i) for i in range(302,501)],
#     5: [Motion_plan_state(-120 + (0.1 * i), -50 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-90 + (0.11 * i), -80 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
#     6: [Motion_plan_state(-85 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-115 - (0.09 * i), -25 - (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
#     7: [Motion_plan_state(-270 + (0.1 * i), 50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-240 - (0.08 * i), 80 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
#     8: [Motion_plan_state(-250 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-280 - (0.1 * i), 105 - (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
#     9: [Motion_plan_state(-260 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-290 + (0.08 * i), 105 + (0.07 * i), traj_time_stamp=i) for i in range(302,501)], 
#     10: [Motion_plan_state(-275 + (0.1 * i), 80 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-245 - (0.13 * i), 50 - (0.12 * i), traj_time_stamp=i) for i in range(302,501)]}
# testing = SharkOccupancyGrid(shark_dict2, 10, boundary_poly, 50, 50)
# boundary_poly = box(0.0, 0.0, 10.0, 10.0)
# shark_dict = {1: [Motion_plan_state(0 + (0.1 * i), 2 + (0.1 * i), traj_time_stamp=0.1*i) for i in range(1,51)]}
# testing = SharkOccupancyGrid(shark_dict, 2, boundary_poly, 2, 4)
# occGrid = testing.constructSharkOccupancyGrid(shark_dict[5])
# auvGrid = testing.constructAUVGrid(occGrid)
# grid = testing.convert()
# plotShark(boundary_poly, shark_dict2)
# print(grid[1])
# with open('AUVGrid_prob_500_turn.csv', 'w', newline='') as csvfile:
#     fieldnames = ['time bin', 'grid']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for time, AUVgrid in grid[1].items():
#         temp = []
#         for _, prob in AUVgrid.items():
#             temp.append(prob)
#         writer.writerow({'time bin': str(time), 'grid': str(temp)})