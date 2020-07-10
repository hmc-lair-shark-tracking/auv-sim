import math
import random
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from shapely.wkt import loads as load_wkt 
from shapely.geometry import Polygon, Point

from gym_rrt.envs.motion_plan_state import Motion_plan_state
from gym_rrt.envs.grid_cell_rrt import Grid_cell_RRT

# from motion_plan_state import Motion_plan_state
# from grid_cell_rrt import Grid_cell_RRT

# TODO: wrong import
# import catalina
# from sharkOccupancyGrid import SharkOccupancyGrid

#from shortest_rrt import Shrt_path

show_animation = True


class Planner_RRT:
    """
    Class for RRT planning
    """
    def __init__(self, start, goal, boundary, obstacles, habitats, exp_rate = 1, dist_to_end = 2, diff_max = 0.5, freq = 50, cell_side_length = 2):
        '''
        Parameters:
            start - initial Motion_plan_state of AUV, [x, y, z, theta, v, w, time_stamp]
            goal - Motion_plan_state of the shark, [x, y, z]
            boundary - max & min Motion_plan_state of the configuration space [[x_min, y_min, z_min],[x_max, y_max, z_max]]
            obstacles - array of Motion_plan_state, representing obstacles [[x1, y1, z1, size1], [x2, y2, z2, size2] ...]
        '''
        # initialize start, goal, obstacle, boundaryfor path planning
        self.start = start
        self.goal = goal

        self.boundary_point = boundary
        self.cell_side_length = cell_side_length

        # discretize the environment into grids
        self.discretize_env(self.cell_side_length)
        
        self.occupied_grid_cells_array = []

        # add the start node to the grid
        self.add_node_to_grid(self.start)
        
        self.obstacle_list = obstacles
        # testing data for habitats
        self.habitats = habitats
        
        # a list of motion_plan_state
        self.mps_list = [self.start]
        self.time_bin = {}

        # if minimum path length is not achieved within maximum iteration, return the latest path
        self.last_path = []

        # setting parameters for path planning
        self.exp_rate = exp_rate
        self.dist_to_end = dist_to_end
        self.diff_max = diff_max
        self.freq = freq

        self.t_start = time.time()


    def discretize_env(self, cell_side_length):
        """
        Separate the environment into grid
        """
        env_btm_left_corner = self.boundary_point[0]
        env_top_right_corner = self.boundary_point[1]
        env_width = env_top_right_corner.x - env_btm_left_corner.x
        env_height = env_top_right_corner.y - env_btm_left_corner.y

        self.env_grid = []

        for row in range(int(env_height) // int(cell_side_length)):
            self.env_grid.append([])
            for col in range(int(env_width) // int(cell_side_length)):
                env_cell_x = env_btm_left_corner.x + col * cell_side_length
                env_cell_y = env_btm_left_corner.y + row * cell_side_length
                self.env_grid[row].append(Grid_cell_RRT(env_cell_x, env_cell_y, side_length = cell_side_length))


    def print_env_grid(self):
        """
        Print the environment grid
            - Each grid cell will take up a line
            - Rows will be separated by ----
        """
        for row in self.env_grid:
            for grid_cell in row:
                print(grid_cell)
            print("----")

    
    def add_node_to_grid(self, mps):
        """

        Parameter:
            mps - a motion plan state object, represent the RRT node that we are trying add to the grid
        """
        hab_index_row = int(mps.y / self.cell_side_length)
        hab_index_col = int(mps.x / self.cell_side_length)

        if hab_index_row >= len(self.env_grid):
            print("auv is out of the habitat environment bound verticaly")
            return
        
        if hab_index_col >= len(self.env_grid[0]):
            print("auv is out of the habitat environment bound horizontally")
            return

        self.env_grid[hab_index_row][hab_index_col].node_list.append(mps)

        # add the grid cell into the occupied grid cell array if it hasn't been added
        if len(self.env_grid[hab_index_row][hab_index_col].node_list) == 1:
            self.occupied_grid_cells_array.append((hab_index_row, hab_index_col))


    def planning(self, max_traj_time = 10.0, max_step = 200, min_length = 250, plan_time=True):
        """
        RRT path planning with a specific goal

        path planning will terminate when:
            1. the path has reached a goal
            2. the maximum planning time has passed

        Parameters:
            start -  an np array
            animation - flag for animation on or off

        """
        # maximum amount for the planner to expand the tree
        t_end = time.time() + max_traj_time

        path = []

        # self.init_live_graph()

        step = 0
    
        for _ in range(max_step):

            # pick the row index and col index for the grid cell where the tree will get expanded
            grid_cell_row, grid_cell_col = random.choice(self.occupied_grid_cells_array)

            done, path = self.generate_one_node(self.env_grid[grid_cell_row][grid_cell_col])

            # if (not done) and path != None:
            #     self.draw_graph(path)
            # elif done:
            #     plt.plot([mps.x for mps in path], [mps.y for mps in path], '-r')
            step += 1

            if done:
                break

        return path, step


    def generate_one_node(self, grid_cell, min_length=250):
        """
        Based on the grid cell, randomly pick a node to expand the tree from from

        Return:
            done - True if we have found a collision-free path from the start to the goal
            path - the collision-free path if there is one, otherwise it's null
            new_node
        """

        # print("chosen grid cell")
        # print(grid_cell)
        # text = input("stop")
        # randomly pick a node from the grid cell
        rand_node = random.choice(grid_cell.node_list)

        new_node = self.steer(rand_node, self.dist_to_end, self.diff_max, self.freq)

        valid_new_node = False
        
        # only add the new node if it's collision free
        if self.check_collision_free(new_node, self.obstacle_list):
            new_node.parent = rand_node
            new_node.length += rand_node.length
            self.mps_list.append(new_node)
            self.add_node_to_grid(new_node)
            valid_new_node = True

        final_node = self.connect_to_goal_curve_alt(self.mps_list[-1], self.exp_rate)

        # if we can create a path between the newly generated node and the goal
        if self.check_collision_free(final_node, self.obstacle_list):
            final_node.parent = self.mps_list[-1]
            path = self.generate_final_course(final_node)   
            return True, path
        
        if valid_new_node:
            return False, new_node
        else:
            return False, None


    def steer(self, mps, dist_to_end, diff_max, freq, velocity=1, traj_time_stamp=False):
        """
        """
        if traj_time_stamp:
            new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, traj_time_stamp=mps.traj_time_stamp)
        else:
            new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, plan_time_stamp=time.time()-self.t_start, traj_time_stamp=mps.traj_time_stamp)

        new_mps.path = [mps]

        n_expand = random.uniform(0, freq)
        n_expand = math.floor(n_expand/1)
        for _ in range(n_expand):
            #setting random parameters
            dist = random.uniform(0, dist_to_end)  # setting random range
            diff = random.uniform(-diff_max, diff_max)  # setting random range

            if abs(dist) > abs(diff):
                s1 = dist + diff
                s2 = dist - diff
                radius = (s1 + s2)/(-s1 + s2)
                phi = (s1 + s2)/ (2 * radius)
                
                ori_theta = new_mps.theta
                new_mps.theta += phi
                delta_x = radius * (math.sin(new_mps.theta) - math.sin(ori_theta))
                delta_y = radius * (-math.cos(new_mps.theta) + math.cos(ori_theta))
                new_mps.x += delta_x
                new_mps.y += delta_y
                if traj_time_stamp:
                    new_mps.traj_time_stamp += (math.sqrt(delta_x ** 2 + delta_y ** 2)) / velocity
                else:
                    new_mps.plan_time_stamp = time.time() - self.t_start
                    new_mps.traj_time_stamp += (math.sqrt(delta_x ** 2 + delta_y ** 2)) / velocity
                new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta=new_mps.theta, traj_time_stamp=new_mps.traj_time_stamp, plan_time_stamp=new_mps.plan_time_stamp))

        new_mps.path[0] = mps

        return new_mps


    def connect_to_goal(self, mps, exp_rate, dist_to_end=float("inf")):
        new_mps = Motion_plan_state(mps.x, mps.y)
        d, theta = self.get_distance_angle(new_mps, self.goal)

        new_mps.path = [new_mps]

        if dist_to_end > d:
            dist_to_end = d

        n_expand = math.floor(dist_to_end / exp_rate)

        for _ in range(n_expand):
            new_mps.x += exp_rate * math.cos(theta)
            new_mps.y += exp_rate * math.sin(theta)
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y))

        d, _ = self.get_distance_angle(new_mps, self.goal)
        if d <= dist_to_end:
            new_mps.path.append(self.goal)
        
        new_mps.path[0] = mps

        return new_mps
    

    def generate_final_course(self, mps):
        path = [mps]
        mps = mps
        while mps.parent is not None:
            reversed_path = reversed(mps.path)
            for point in reversed_path:
                path.append(point)
            mps = mps.parent
        #path.append(mps)

        return path

    
    def get_random_mps_from_region(self, region, size_max = 15):
        """
        Generate a random size based on the region picked by the neural network
        """
        x_min = region.x
        x_max = region.x + region.size
        y_min = region.y
        y_max = region.x + region.size
        
        ran_x = random.uniform(x_min, x_max)
        ran_y = random.uniform(y_min, y_max)
        ran_theta = random.uniform(-math.pi, math.pi)
        ran_size = random.uniform(1, size_max)

        return Motion_plan_state(ran_x, ran_y, theta=ran_theta, size=ran_size)

    
    def init_live_graph(self):
        _, self.ax = plt.subplots()

        for row in self.env_grid:
            for grid_cell in row:
                cell = Rectangle((grid_cell.x, grid_cell.y), width=grid_cell.side_length, height=grid_cell.side_length, color='#2a753e', fill=False)
                self.ax.add_patch(cell)
        
        for obstacle in self.obstacle_list:
            self.plot_circle(obstacle.x, obstacle.y, obstacle.size)

        self.ax.plot(self.start.x, self.start.y, "xr")
        self.ax.plot(self.goal.x, self.goal.y, "xr")

    
    def draw_graph(self, rnd=None):
        # plt.clf()  # if we want to clear the plot
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd != None:
            # self.ax.plot(rnd.x, rnd.y, ",", color="#000000")

            self.ax.plot([point.x for point in rnd.path], [point.y for point in rnd.path], '-', color="#000000")
        else:
            for mps in self.mps_list:
                if mps.parent:
                    plt.plot([point.x for point in mps.path], [point.y for point in mps.path], '-g')

        plt.axis("equal")

        plt.pause(1)


    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)
    

    def connect_to_goal_curve(self, mps1):
        a = (self.goal.y - mps1.y)/(np.cosh(self.goal.x) - np.cosh(mps1.x))
        b = mps1.y - a * np.cosh(mps1.x)
        x = np.linspace(mps1.x, self.goal.x, 15)
        x = x.tolist()
        y = a * np.cosh(x) + b
        y = y.tolist()

        new_mps = Motion_plan_state(x[-2], y[-2])
        new_mps.path.append(mps1)
        for i in range(1, len(x)):
            new_mps.path.append(Motion_plan_state(x[i], y[i]))
            new_mps.length += math.sqrt((new_mps.path[i].x-new_mps.path[i-1].x) ** 2 +  (new_mps.path[i].y-new_mps.path[i-1].y) ** 2)
        '''plt.plot(mps1.x, mps1.y, color)
        plt.plot(self.goal.x, self.goal.y, color)
        plt.plot([mps.x for mps in new_mps.path], [mps.y for mps in new_mps.path], color)
        plt.show()'''
        return new_mps
    
    
    def connect_to_goal_curve_alt(self, mps, exp_rate):
        new_mps = Motion_plan_state(mps.x, mps.y, theta=mps.theta, traj_time_stamp=mps.traj_time_stamp)
        theta_0 = new_mps.theta
        _, theta = self.get_distance_angle(mps, self.goal)
        diff = theta - theta_0
        diff = self.angle_wrap(diff)
        
        if abs(diff) > math.pi / 2:
            return

        #polar coordinate
        r_G = math.hypot(self.goal.x - new_mps.x, self.goal.y - new_mps.y)
        phi_G = math.atan2(self.goal.y - new_mps.y, self.goal.x - new_mps.x)


        # arc
        if phi_G - new_mps.theta != 0:
            phi = 2 * self.angle_wrap(phi_G - new_mps.theta)
            # prevent a dividing by 0 error
        else:
            return
        
        if math.sin(phi_G - new_mps.theta) != 0:
            radius = r_G / (2 * math.sin(phi_G - new_mps.theta))
        else:
            return

        length = radius * phi
        if phi > math.pi:
            phi -= 2 * math.pi
            length = -radius * phi
        elif phi < -math.pi:
            phi += 2 * math.pi
            length = -radius * phi
        new_mps.length += length

        ang_vel = phi / (length / exp_rate)

        #center of rotation
        x_C = new_mps.x - radius * math.sin(new_mps.theta)
        y_C = new_mps.y + radius * math.cos(new_mps.theta)

        n_expand = math.floor(length / exp_rate)
        for i in range(n_expand+1):
            new_mps.x = x_C + radius * math.sin(ang_vel * i + theta_0)
            new_mps.y = y_C - radius * math.cos(ang_vel * i + theta_0)
            new_mps.theta = ang_vel * i + theta_0
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta = new_mps.theta, plan_time_stamp=time.time()-self.t_start))
        
        return new_mps

    def angle_wrap(self, ang):
        if -math.pi <= ang <= math.pi:
            return ang
        elif ang > math.pi: 
            ang += (-2 * math.pi)
            return self.angle_wrap(ang)
        elif ang < -math.pi: 
            ang += (2 * math.pi)
            return self.angle_wrap(ang)

    def get_closest_mps(self, ran_mps, mps_list):
        min_dist, _ = self.get_distance_angle(mps_list[0], ran_mps)
        closest_mps = mps_list[0]
        for mps in mps_list:
            dist, _ = self.get_distance_angle(mps, ran_mps)
            if dist < min_dist:
                min_dist = dist
                closest_mps = mps
        return closest_mps
    
    def get_closest_mps_time(self, ran_time, mps_list):
        while len(mps_list) > 3:
            left = mps_list[len(mps_list)//2 - 1]
            left_diff = abs(left.plan_time_stamp - ran_time)
            right = mps_list[len(mps_list)//2 + 1]
            right_diff = abs(right.plan_time_stamp - ran_time)

            index =  len(mps_list)//2 
            if left_diff >= right_diff:
                mps_list = mps_list[index:]
            else:
                mps_list = mps_list[:index]
        
        return mps_list[0]

    def check_collision_free(self, mps, obstacleList):
        """
        Collision
        Return:
            True -  if the new node (as a motion plan state) and its path is collision free
            False - otherwise
        """
        if mps is None:
            return False

        dList = []
        for obstacle in obstacleList:
            for point in mps.path:
               d, _ = self.get_distance_angle(obstacle, point)
               dList.append(d) 

            if min(dList) <= obstacle.size:
                return False  # collision
    
        for point in mps.path:
            if not self.check_within_boundary(point):
                return False

        return True  # safe
    

    def check_collision_obstacle(self, mps, obstacleList):
        for obstacle in obstacleList:
            d, _ = self.get_distance_angle(obstacle, mps)
            if d <= obstacle.size:
                return False
        return True


    def check_within_boundary(self, mps):
        """
        Warning: 
            For a rectangular environment only

        Return:
            True - if it's within the environment boundary
            False - otherwise
        """
        env_btm_left_corner = self.boundary_point[0]
        env_top_right_corner = self.boundary_point[1]

        within_x_bound = (mps.x >= env_btm_left_corner.x) and (mps.x <= env_top_right_corner.x)
        within_y_bound = (mps.y >= env_btm_left_corner.y) and (mps.y <= env_top_right_corner.y)

        return (within_x_bound and within_y_bound)

    def get_distance_angle(self, start_mps, end_mps):
        """
        Return
            - the range and 
            - the bearing between 2 points, represented as 2 Motion_plan_states
        """
        dx = end_mps.x-start_mps.x
        dy = end_mps.y-start_mps.y
        #dz = end_mps.z-start_mps.z
        dist = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy,dx)
        return dist, theta
    

    def cal_length(self, path):
        length = 0
        for i in range(1, len(path)):
            length += math.sqrt((path[i].x-path[i-1].x)**2 + (path[i].y-path[i-1].y)**2)
        return length
    

    def cal_boundary_peri(self):
        peri = 0
        for i in range(len(self.boundary)-1):
            dist, _ = self.get_distance_angle(self.boundary[i], self.boundary[i+1])
            peri += dist
        
        return peri
    
    def plot_performance(self, time_list, perf_list):
        _, ax = plt.subplots()
        ax.plot(time_list, perf_list)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('optimal sum cost')
        ax.set_title('RRT performance')
        ax.legend()

        plt.show()

def main():
    auv_init_pos = Motion_plan_state(x = 10.0, y = 10.0, z = -5.0, theta = 0.0)
    shark_init_pos = Motion_plan_state(x = 35.0, y = 40.0, z = -5.0, theta = 0.0)
    
    obstacle_array = [\
        Motion_plan_state(x=12.0, y=38.0, size=4),\
        Motion_plan_state(x=17.0, y=34.0, size=5),\
        Motion_plan_state(x=20.0, y=29.0, size=4),\
        Motion_plan_state(x=25.0, y=25.0, size=3),\
        Motion_plan_state(x=29.0, y=20.0, size=4),\
        Motion_plan_state(x=34.0, y=17.0, size=3),\
        Motion_plan_state(x=37.0, y=8.0, size=5)\
    ]

    boundary_array = [Motion_plan_state(x=0.0, y=0.0), Motion_plan_state(x = 50.0, y = 50.0)]

    step_array = []
    success_count = 0
    for _ in range(1000):
        rrt = Planner_RRT(auv_init_pos, shark_init_pos, boundary_array, obstacle_array, [], freq=10, cell_side_length=5)
        path, step = rrt.planning(max_step=300)
        if path != [] and type(path) == list:
            success_count += 1
        step_array.append(step)

    print(step_array)
    print(np.mean(step_array))
    print("success")
    print(success_count)
    text = input("stop")

    if path != [] and type(path) == list:
        rrt.draw_graph()
        plt.plot([mps.x for mps in path], [mps.y for mps in path], '-r')
    else:
        print("Failed to find path")
        rrt.draw_graph()

    plt.show()
        


if __name__ == '__main__':
    main()