import math
import random

import matplotlib.pyplot as plt
import numpy as np

from motion_plan_state import Motion_plan_state

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

    def planning(self, max_iter = 500, exp_rate = 0.5, dist_to_end = 3, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.mps_list = [self.start]
        for i in range(max_iter):
            ran_mps = self.get_random_mps()
            closest_mps = self.get_closest_mps(ran_mps, self.mps_list)

            new_mps = self.steer(closest_mps, ran_mps, exp_rate, dist_to_end)

            if self.check_collision(new_mps, self.obstacle_list):
                new_mps.parent = closest_mps
                self.mps_list.append(new_mps)

            if animation and i % 5 == 0:
                self.draw_graph(ran_mps)
            
            d, _ = self.get_distance_angle(self.mps_list[-1], self.goal)
            if d <= dist_to_end:
                final_mps = self.steer(self.mps_list[-1], self.goal, exp_rate, dist_to_end)
                if self.check_collision(final_mps, self.obstacle_list):
                    return self.generate_final_course(len(self.mps_list) - 1)

            if animation and i % 5:
                self.draw_graph(ran_mps)

        return None  # cannot find path

    def steer(self, from_mps, to_mps, exp_rate, dist_to_end=float("inf")):
        new_mps = Motion_plan_state(from_mps.x, from_mps.y)
        d, theta = self.get_distance_angle(new_mps, to_mps)

        new_mps.path = [new_mps]

        if dist_to_end > d:
            dist_to_end = d

        n_expand = math.floor(dist_to_end / exp_rate)

        for _ in range(n_expand):
        #while d > dist_to_end:
            new_mps.x += exp_rate * math.cos(theta)
            new_mps.y += exp_rate * math.sin(theta)
            new_mps.path.append(Motion_plan_state(new_mps.x, new_mps.y))

        d, _ = self.get_distance_angle(new_mps, to_mps)
        if d <= dist_to_end:
            new_mps.path.append(to_mps)

        #new_node.parent = from_node
        new_mps.path[0] = from_mps

        return new_mps

    def generate_final_course(self, goal_index):
        path = [self.goal]
        mps = self.mps_list[goal_index]
        while mps.parent is not None:
            path.append(mps)
            mps = mps.parent
        path.append(mps)

        return path

    def get_random_mps(self):
        ran_x = random.uniform(self.min_area.x, self.max_area.x)
        ran_y = random.uniform(self.min_area.y, self.max_area.y)
        #ran_z = random.uniform(self.min_area.z, self.max_area.z)
        mps = Motion_plan_state(ran_x, ran_y)
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
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

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

        return True  # safe

    def get_distance_angle(self, start_mps, end_mps):
        dx = end_mps.x-start_mps.x
        dy = end_mps.y-start_mps.y
        #dz = end_mps.z-start_mps.z
        dist = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy,dx)
        return dist, theta


def main():
    initial = Motion_plan_state(0,0)
    goal = Motion_plan_state(6,10)
    obstacle_list = [Motion_plan_state(5,5,size=1),Motion_plan_state(3,6,size=2),Motion_plan_state(3,8,size=2),\
    Motion_plan_state(3,10,size=2),Motion_plan_state(7,5,size=2),Motion_plan_state(9,5,size=2),Motion_plan_state(8,10,size=1)]
    boundary = [Motion_plan_state(0,0), Motion_plan_state(15,15)]
    rrt = RRT(initial, goal, obstacle_list, boundary)
    path = rrt.planning(animation=False)
    print(path)

    # Draw final path
    if show_animation:
        rrt.draw_graph()
        plt.plot([mps.x for mps in path], [mps.y for mps in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()