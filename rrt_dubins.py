import math
import random

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from motion_plan_state import Motion_plan_state
#from shortest_rrt import Shrt_path

show_animation = True


class RRT:
    """
    Class for RRT planning
    """
    def __init__(self, initial_location, goal_location, obstacle_list, boundary):
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

    def planning(self, max_iter = 1000, exp_rate = 1, dist_to_end = 1, diff_max = 0.5, freq = 10, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.mps_list = [self.start]
        for i in range(max_iter):
            ran_mps = self.get_random_mps()
            closest_mps = self.get_closest_mps(ran_mps, self.mps_list)

            new_mps = self.steer(closest_mps, dist_to_end, diff_max, freq)

            if self.check_collision(new_mps, self.obstacle_list):
                new_mps.parent = closest_mps
                new_mps.length += closest_mps.length
                self.mps_list.append(new_mps)

            if animation and i % 5 == 0:
                self.draw_graph(ran_mps)
                
            final_mps = self.connect_to_goal_curve_alt(self.mps_list[-1], exp_rate)
            if self.check_collision(final_mps, self.obstacle_list):
                final_mps.parent = self.mps_list[-1]
                if self.mps_list[-1].parent is None:
                    return [final_mps.length, self.generate_final_course(final_mps)]
                final_mps.length += self.mps_list[-1].parent.length
                return [final_mps.length, self.generate_final_course(final_mps)]

            if animation and i % 5:
                self.draw_graph(ran_mps)

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

        new_mps = Motion_plan_state(mps.x, mps.y, theta = mps.theta)

        new_mps.path = [mps]

        '''if dist_to_end > d:
            dist_to_end = dist_to_end / 10

        n_expand = math.floor(dist_to_end / exp_rate)'''

        n_expand = random.uniform(0, freq)
        n_expand = math.floor(n_expand/1)
        for _ in range(n_expand):
        #while new_mps.x > self.min_area.x and new_mps.x < self.max_area.x and new_mps.y > self.min_area.y and new_mps.y < self.max_area.y:
        #while d > dist_to_end:

            #setting random parameters
            dist = random.uniform(0, dist_to_end)#setting random range
            diff = random.uniform(-diff_max, diff_max)#setting random range

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
                new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta=new_mps.theta))
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
        path = [self.goal]
        mps = mps
        while mps.parent is not None:
            reversed_path = reversed(mps.path)
            for point in reversed_path:
                path.append(point)
            mps = mps.parent
        #path.append(mps)

        return path

    def get_random_mps(self):
        ran_x = random.uniform(self.min_area.x, self.max_area.x)
        ran_y = random.uniform(self.min_area.y, self.max_area.y)
        ran_theta = random.uniform(-math.pi, math.pi)
        #ran_z = random.uniform(self.min_area.z, self.max_area.z)
        mps = Motion_plan_state(ran_x, ran_y, theta=ran_theta)
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

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
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
        new_mps = Motion_plan_state(mps.x, mps.y, theta=mps.theta)
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
            new_mps.x = x_C + radius * math.sin(ang_vel * i + theta_0)
            new_mps.y = y_C - radius * math.cos(ang_vel * i + theta_0)
            new_mps.theta = ang_vel * i + theta_0
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y, theta = new_mps.theta))
        
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

def main():
    start = Motion_plan_state(0,0)
    goal = Motion_plan_state(7,4)
    boundary = [Motion_plan_state(0,0), Motion_plan_state(10,10)]
    obstacle_array = [Motion_plan_state(5,7, size=2),Motion_plan_state(4,2, size=1)]
    rrt = RRT(start, goal, obstacle_array, boundary)
    path = rrt.planning(animation=False)
    print(path)

    # Draw final path
    if path is not None:
        if show_animation:
            rrt.draw_graph()
            plt.plot([mps.x for mps in path[1]], [mps.y for mps in path[1]], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main()