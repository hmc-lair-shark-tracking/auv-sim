import math
import random
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from motion_plan_state import Motion_plan_state
from cost import Cost
#from shortest_rrt import Shrt_path

show_animation = True


class RRT:
    """
    Class for RRT planning
    """
    def __init__(self, initial_location, goal_location, obstacle_list, boundary, exp_rate = 1, dist_to_end = 1, diff_max = 0.5, freq = 20):
        '''setting parameters:
            initial_location: initial Motion_plan_state of AUV, [x, y, z, theta, v, w, time_stamp]
            goal_location: Motion_plan_state of the shark, [x, y, z]
            obstacle_list: Motion_plan_state of obstacles [[x1, y1, z1, size1], [x2, y2, z2, size2] ...]
            boundary: max & min Motion_plan_state of the configuration space [[x_min, y_min, z_min],[x_max, y_max, z_max]]'''
        self.start = initial_location
        self.goal = goal_location
        self.min_area = boundary[0]
        self.max_area = boundary[1]
        self.obstacle_list = obstacle_list
        self.mps_list = [] # a list of motion_plan_state
        self.habitats = []

        # keep track of the current time that we are in
        # each iteration in the while loop will be assumed as 0.1 sec
        self.curr_time = 0
        self.plan_time_interval = 0.001

        #if minimum path length is not achieved within maximum iteration, return the latest path
        self.last_path = []

        #setting parameters for path planning
        self.exp_rate = exp_rate
        self.dist_to_end = dist_to_end
        self.diff_max = diff_max
        self.freq = freq

    def exploring(self, habitats, plot_interval, test_time=5.0, plan_time=True, weights=[1,-1,-1]):
        """
        rrt path planning without setting a specific goal, rather try to explore the configuration space as much as possible
        calculate cost while expand and keep track of the current optimal cost path

        max_iter: maximum iteration for the tree to expand
        plan_time: expand by randomly picking a time stamp and find the motion_plan_state along the path with smallest time difference
        """
        self.habitats = habitats

        #keep track of the motion_plan_state whose path is optimal
        opt_cost = [float("inf")]
        opt_path = None
        opt_cost_list = []

        #keep track of the longest single path in the tree to normalize every path length
        peri_boundary = 2 * (self.max_area.x - self.min_area.x) + 2 * (self.max_area.y - self.min_area.y)

        #initialize cost function
        cal_cost = Cost()

        self.mps_list = [self.start]

        self.t_start = time.time()
        n_expand = math.ceil(test_time / plot_interval)
        for i in range(1, n_expand + 1):
            t_end = self.t_start + i * plot_interval
            while time.time() < t_end:
                #find the closest motion_plan_state by generating a random time stamp and 
                #find the motion_plan_state whose time stamp is closest to it
                if plan_time:
                    ran_time = random.uniform(0, test_time * self.freq)
                    closest_mps = self.get_closest_mps_time(ran_time, self.mps_list)
                #find the closest motion_plan_state by generating a random motion_plan_state
                #and find the motion_plan_state with smallest distance
                else:
                    ran_mps = self.get_random_mps()
                    closest_mps = self.get_closest_mps(ran_mps, self.mps_list)
                
                new_mps = self.steer(closest_mps, self.dist_to_end, self.diff_max, self.freq)

                if self.check_collision(new_mps, self.obstacle_list):
                    new_mps.parent = closest_mps
                    path = self.generate_final_course(new_mps)
                    new_mps.length = self.cal_length(path)
                    self.mps_list.append(new_mps)
                    #Question: how to normalize the path length?
                    if new_mps.length != 0:
                        cost = cal_cost.habitat_time_cost_func(path, new_mps.length, self.habitats, peri_boundary, weights=weights)
                        if cost[0] < opt_cost[0]:
                            opt_cost = cost
                            opt_path = [new_mps.length, path]
                
            opt_cost_list.append(opt_cost[0])

        #self.plot_performance(time_list, opt_cost_list)
        return {"path length": opt_path[0], "path": opt_path[1], "cost": opt_cost, "cost list": opt_cost_list}
        

    def planning(self, max_iter = 1000, animation=False, min_length = 250, plan_time=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.mps_list = [self.start]
        for i in range(max_iter):
            #find the closest motion_plan_state by generating a random time stamp and 
            #find the motion_plan_state whose time stamp is closest to it
            if plan_time:
                ran_time = random.uniform(0, self.curr_time)
                closest_mps = self.get_closest_mps_time(ran_time, self.mps_list)
            #find the closest motion_plan_state by generating a random motion_plan_state
            #and find the motion_plan_state with smallest distance
            else:
                ran_mps = self.get_random_mps()
                closest_mps = self.get_closest_mps(ran_mps, self.mps_list)
            
            new_mps = self.steer(closest_mps, self.dist_to_end, self.diff_max, self.freq)

            if self.check_collision(new_mps, self.obstacle_list):
                new_mps.parent = closest_mps
                new_mps.length += closest_mps.length
                self.mps_list.append(new_mps)

            if animation and i % 5 == 0:
                self.draw_graph(closest_mps)
                
            final_mps = self.connect_to_goal_curve_alt(self.mps_list[-1], self.exp_rate)
            if self.check_collision(final_mps, self.obstacle_list):
                final_mps.parent = self.mps_list[-1]
                path = self.generate_final_course(final_mps)
                final_mps.length = self.cal_length(path)
                if final_mps.length >= min_length:
                    return [final_mps.length, path]
                else:
                    self.last_path = [final_mps.length, path]

            if animation and i % 5:
                self.draw_graph(closest_mps)

        #return the latest possible path
        if self.last_path != []:
            return self.last_path
        
        return None  # cannot find path

    def steer(self, mps, dist_to_end, diff_max, freq):
        #dubins library
        '''new_mps = Motion_plan_state(from_mps.x, from_mps.y, theta = from_mps.theta)
        new_mps.path = []

        q0 = (from_mps.x, from_mps.y, from_mps.theta)
        q1 = (to_mps.x, to_mps.y, to_mps.theta)
        turning_radius = 1.0

        path = dubins.shortest_path(q0, q1, turning_radius)
        configurations, _ = path.sample_many(exp_rate)
        for configuration in configurations:
            new_mps.path.append(Motion_plan_state(x = configuration[0], y = configuration[1], theta = configuration[2]))

        new_mps.path.append(to_mps)

        dubins_path = new_mps.path
        new_mps = dubins_path[-2]
        new_mps.path = dubins_path'''

        new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta, time_stamp=self.curr_time)

        new_mps.path = [mps]

        '''if dist_to_end > d:
            dist_to_end = dist_to_end / 10

        n_expand = math.floor(dist_to_end / exp_rate)'''

        n_expand = random.uniform(0, freq)
        n_expand = math.floor(n_expand/1)
        for _ in range(n_expand):
            #setting random parameters
            dist = random.uniform(0, dist_to_end)#setting random range
            diff = random.uniform(-diff_max, diff_max)#setting random range

            if abs(dist) > abs(diff):
                #update time stamp
                self.curr_time += self.plan_time_interval

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
                new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta=new_mps.theta, time_stamp=time.time()-self.t_start))
                new_mps.length += math.sqrt(delta_x ** 2 + delta_y ** 2)

            #d, theta = self.get_distance_angle(new_mps, to_mps)

        '''d, _ = self.get_distance_angle(new_mps, to_mps)
        if d <= dist_to_end:
            new_mps.path.append(to_mps)'''

        #new_node.parent = from_node
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

    def get_random_mps(self, size_max=15):
        ran_x = random.uniform(self.min_area.x, self.max_area.x)
        ran_y = random.uniform(self.min_area.y, self.max_area.y)
        ran_theta = random.uniform(-math.pi, math.pi)
        ran_size = random.uniform(0, size_max)
        mps = Motion_plan_state(ran_x, ran_y, theta=ran_theta, size=ran_size)
        #ran_z = random.uniform(self.min_area.z, self.max_area.z)
        
        return mps

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for mps in self.mps_list:
            if mps.parent:
                plt.plot([point.x for point in mps.path], [point.y for point in mps.path], '-g')

        for obstacle in self.obstacle_list:
            self.plot_circle(obstacle.x, obstacle.y, obstacle.size)

        for habitat in self.habitats:
            self.plot_circle(habitat.x, habitat.y, habitat.size, color="-r")

        plt.plot(self.start.x, self.start.y, "xr")
        #plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_area.x, self.max_area.x, self.min_area.y, self.max_area.y])
        plt.grid(True)
        plt.pause(0.01)

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
        new_mps = Motion_plan_state(mps.x, mps.y, theta=mps.theta, time_stamp=mps.time_stamp)
        theta_0 = new_mps.theta
        _, theta = self.get_distance_angle(mps, self.goal)
        diff = theta - theta_0
        diff = self.angle_wrap(diff)
        if abs(diff) > math.pi / 2:
            return

        #polar coordinate
        r_G = math.hypot(self.goal.x - new_mps.x, self.goal.y - new_mps.y)
        phi_G = math.atan2(self.goal.y - new_mps.y, self.goal.x - new_mps.x)

        #arc
        phi = 2 * self.angle_wrap(phi_G - new_mps.theta)
        radius = r_G / (2 * math.sin(phi_G - new_mps.theta))

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
            #update time stamp
            self.curr_time += self.plan_time_interval
            new_mps.x = x_C + radius * math.sin(ang_vel * i + theta_0)
            new_mps.y = y_C - radius * math.cos(ang_vel * i + theta_0)
            new_mps.theta = ang_vel * i + theta_0
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta = new_mps.theta, time_stamp=self.curr_time))
        
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
        min_dist, _ = self.get_distance_angle(mps_list[0],ran_mps)
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
            left_diff = abs(left.time_stamp - ran_time)
            right = mps_list[len(mps_list)//2 + 1]
            right_diff = abs(right.time_stamp - ran_time)

            index =  len(mps_list)//2 
            if left_diff >= right_diff:
                mps_list = mps_list[index:]
            else:
                mps_list = mps_list[:index]
        
        return mps_list[0]

    def check_collision(self, mps, obstacleList):

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
            if point.x < self.min_area.x or point.x > self.max_area.x \
                or point.y < self.min_area.y or point.y > self.max_area.y:
                return False

        return True  # safe
    
    def check_collision_obstacle(self, mps, obstacleList):
        for obstacle in obstacleList:
            d, _ = self.get_distance_angle(obstacle, mps)
            if d <= obstacle.size:
                return False
        
        return True

    def get_distance_angle(self, start_mps, end_mps):
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
    
    def plot_performance(self, time_list, perf_list):
        fig, ax = plt.subplots()
        ax.plot(time_list, perf_list)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('optimal sum cost')
        ax.set_title('RRT performance')
        ax.legend()

        plt.show()

def main():
    start = Motion_plan_state(10,15)
    goal = Motion_plan_state(7,4)
    boundary = [Motion_plan_state(0,0), Motion_plan_state(100,100)]
    obstacle_array = [Motion_plan_state(5,7, size=5),Motion_plan_state(14,22, size=3)]
    habitats = [Motion_plan_state(63,23, size=5), Motion_plan_state(12,45,size=7), Motion_plan_state(51,36,size=5), Motion_plan_state(45,82,size=5),\
        Motion_plan_state(60,65,size=10), Motion_plan_state(80,79,size=5),Motion_plan_state(85,25,size=6)]
    rrt = RRT(start, goal, obstacle_array, boundary)
    #path = rrt.planning(animation=False, min_length=0)
    path = rrt.exploring(habitats, 0.5, test_time=5.0, plan_time=True, weights=[1,-4.5,-4.5])
    print(path["cost"])
    
    # Draw final path
    if path is not None:
        plt.figure(1)  
        rrt.draw_graph()
        plt.plot([mps.x for mps in path["path"]], [mps.y for mps in path["path"]], '-r')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()