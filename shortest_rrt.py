import math
import random
import time
import threading

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import statistics

from motion_plan_state import Motion_plan_state
from rrt_dubins import RRT

show_animation = False

class Performance:
    #class to analyze performance of motion planning algorithms for different configuration spaces
    def __init__(self, obstacle_list, boundary, plan_time = 5, plot_interval = 1, num_space = 10):
        self.plan_time = plan_time #time limit for motion planning algorithm to run for each configuration space
        self.plot_interval = plot_interval #time interval to calculate the shortest length
        self.bin_list = []
        self.num_space = num_space
        self.boundary = boundary
        self.obstacle_list = obstacle_list
        self.paths = []
        self.mean_list = []

    class Planning():
        #class for the performance of motion planning algorithms for a single configuration space
        def __init__(self, plan_time, plot_interval):
            self.plan_time = plan_time
            self.plot_interval = plot_interval
            self.shortest_length = float("inf") #length of the shortest path, default to be infinite long
            self.shortest_path = []             #the shortest path generated regarding this configuration space
            self.time_list = [] 
            self.shrtpath_list = []             #a list of the shortest length at each plot_interval, len() == plan_time / plot_interval

        def find_shortest_path(self, planner, initial, goal, obstacle_list, boundary):
            n_expand = math.ceil(self.plan_time / self.plot_interval)
            ori_t = time.time()

            for i in range(1, n_expand+1):

                t_end = ori_t + i * self.plot_interval

                while time.time() < t_end:
                    path_planning = RRT(initial, goal, obstacle_list, boundary)
                    result = path_planning.planning()
                    if result is not None:
                        self.get_shortest(result)

                    #self.plot_perf()
                
                self.shrtpath_list.append(self.shortest_length)
                self.time_list.append(time.time() - ori_t)
                self.plot_perf()

            #return self.shortest_length, self.shortest_path
            return self.shrtpath_list
    
        def get_shortest(self, result):
            length, path = result[0], result[1]
            if length < self.shortest_length:
                self.shortest_length = length
                self.shortest_path = path
    
        def plot_perf(self, show = False):
            plt.figure(2)
            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            plt.title("RRT performance: shortest length as a function of time")
            plt.xlabel("plan time/s")
            plt.ylabel("shortest length")
            plt.xlim(1, math.ceil(self.plan_time / self.plot_interval))
            #plt.xticks(self.bin_list)
            plt.plot(self.time_list, self.shrtpath_list, color="r")

            plt.pause(0.01)
            if show:
                plt.show()
    
    def performing(self):
        rrt = RRT(None, None, self.obstacle_list, self.boundary)

        n_expand = math.ceil(self.plan_time / self.plot_interval)
        self.bin_list = [self.plot_interval * i for i in range(1, n_expand+1)]
        self.paths = [[] for i in range(n_expand)]

        for i in range(self.num_space):
            initial = rrt.get_random_mps()
            while not rrt.check_collision_obstacle(initial, self.obstacle_list):
                initial = rrt.get_random_mps()
            goal = rrt.get_random_mps()
            while not rrt.check_collision_obstacle(goal, self.obstacle_list):
                goal = rrt.get_random_mps()

            testing = self.Planning(self.plan_time, self.plot_interval)
            path_list = testing.find_shortest_path(RRT, initial, goal, self.obstacle_list, self.boundary)
            for i in range(len(path_list)):
                if path_list[i] == float("inf"):
                    self.paths[i].append(0)
                else:
                    self.paths[i].append(path_list[i])
            
            print(initial, goal, path_list)
        self.cal_performance()
        self.plot_performance(show = True)

        return self.mean_list[-1]

    def cal_performance(self):
        for path in self.paths:
            self.mean_list.append(statistics.mean(path))
    
    def plot_performance(self, show = False):
        plt.figure(3)
        plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
        plt.title("RRT performance")
        plt.xlabel("plan time/s")
        plt.ylabel("average shortest length")
        plt.xticks(self.bin_list)
        plt.bar(self.bin_list, self.mean_list, color="g")

        if show:
            plt.show()

def main():
    obstacle_list = [Motion_plan_state(5,5,size=1),Motion_plan_state(3,6,size=2),Motion_plan_state(3,8,size=2),\
    Motion_plan_state(3,10,size=2),Motion_plan_state(7,5,size=2),Motion_plan_state(9,5,size=2),Motion_plan_state(8,10,size=1)]
    boundary = [Motion_plan_state(0,0), Motion_plan_state(15,15)]

    testing1 = Performance(obstacle_list, boundary)
    length = testing1.performing()
    #print("the average shortest length is :" + str(length))

if __name__ == '__main__':
    main()