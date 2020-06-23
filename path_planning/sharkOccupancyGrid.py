import math
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point

import catalina
from motion_plan_state import Motion_plan_state

class SharkOccupancyGrid:
    '''
    a class to construct shark occupancy grid maps in each time bin to represent the distribution of shark positions 
    
    the map will be pre-constructed before planning starts using shark trajectories
    each time cost function involving keeping track of sharks in sonar range is calculated, corresponding occupancy 
        grid will be used
    '''

    def __init__(self, shark_dict, cell_size, boundary, bin_interval):
        '''
        create a occupancy dictionary to store occupancy grid maps at different time bins
        key is time bin, value is the corresponding occupancy grid map

        paramaters:
            shark_dict: a dictionary representing shark trajectories, key is shark_id and value is the corresponding trajectory
            cell_size: cell size in meters
            boundary: a Polygon object representing the configuration space, needed to be splitted into cells
            bin_interval: time interval to construct a separate shark occupancy grid, in seconds
        the probability of occupancy of each cell is calculated by the time sharks spent in the cell / duration of time bin
        '''
        self.data = shark_dict
        self.cell_size = cell_size
        self.cell_list = self.splitCell(boundary)
        self.bin_interval = bin_interval

        self.bin_list = []
    
    def convert(self):
        '''
        convert a dictionary of shark trajectories
            key: shark ID, int
            value: shark trajectory, a list of Motion_plan_state
        
        output: a dictionary of shark occupancy grid maps
            key: timebin, tuple(start time, end time)
            value: shark occupancy grid during this timebin
        '''
        longest_time = 0
        for _, traj in self.data.items():
            if traj[-1].traj_time_stamp > longest_time:
                longest_time = traj[-1].traj_time_stamp
        self.bin_list = self.createBinList(longest_time)

        #convert to a dictionary, whose key is time bin
        timeBinDict = self.convertToTimeBin()

        result = {}
        for time, traj_dict in timeBinDict.items():
            grid = self.constrcutGrid(traj_dict)
            result[time] = grid

        return result
    
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
    
    def constrcutGrid(self, shark_traj_dict):
        '''
        create a single shark occupancy grid at during one time bin for multiple sharks
        the occupancy probability of each cell should reflect both time sharks spent in this cell 
            and the number of sharks in this cell
        
        parameters:
            shark_traj_dict: a dictionary stored separate shark trajectories within the single time bin
        output: a dictionary representing the occupancy of each cell
            key: Polygon object representing each cell
            value: occupancy probability of the cell, calculated by num of sharks * average time sharks spent in this cell
        '''
        #initialize occupancy grid, initial occupancy set to 0
        occupancyGrid = {}
        for cell in self.cell_list:
            occupancyGrid[cell.bounds] = 0
        
        #calculate occupancy
        #Question: need to distinguish different sharks?
        for shark, traj in shark_traj_dict.items():
            for point in traj:
                point = Point(point.x, point.y)
                for cell in self.cell_list:
                    if point.within(cell) or cell.touches(point):
                        occupancyGrid[cell.bounds] += round(1 / len(traj), 3)
                        break
        
        return occupancyGrid
    
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
    
    def createBinList(self, timeLength):
        '''
        create a list of tuples representing each time bin
        '''
        n_expand = math.floor(timeLength / self.bin_interval)
        bin_list = []
        for i in range(n_expand):
            bin_list.append((i * self.bin_interval, (i + 1) * self.bin_interval))
        
        return bin_list


# boundary = []
# for b in catalina.BOUNDARIES:
#     pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
#     boundary.append((pos[0], pos[1]))
boundary_poly = Polygon([(0,0), (0,100), (100, 0)])
shark_dict = {1: [Motion_plan_state(0 + (1 * i), 6 + (2 * i), traj_time_stamp=0.1*i) for i in range(1,201)], 
    2: [Motion_plan_state(80 - (1.5 * i), 95 - (1 * i), traj_time_stamp=0.1*i) for i in range(1,201)]}
testing = SharkOccupancyGrid(shark_dict, 10, boundary_poly, 5)
print(testing.convert())