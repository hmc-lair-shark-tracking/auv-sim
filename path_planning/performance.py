import math
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import random

from rrt_dubins import RRT, createSharkGrid
from motion_plan_state import Motion_plan_state
import catalina
from sharkOccupancyGrid import SharkOccupancyGrid, splitCell

def summary_1(weight, obstacles, boundary, habitats, shark_dict, sharkGrid, test_num=100):
    '''
        generate a summary about each term of one specific cost function, given randomly chosen environment
    
        cost_func: a list of lists of weights assigned to each term in the cost function
        test_num: the number of tests to run under a specific cost function

        output:
        cost_avr: a dictionary summarizing the result of each term of the cost function, 
            key will be weight i.e. w1, w2, ...
            value will be the average cost of each term
    '''
    testing = RRT(boundary, obstacles, shark_dict, sharkGrid)

    cost_summary_ex = [[] for _ in range(len(weight))]
    cost_summary_rp = [[] for _ in range(len(weight))]

    for _ in range(test_num):
        initial_x = random.uniform(-300, -100)
        initial_y = random.uniform(-100, 100)
        initial = Point(initial_x, initial_y)
        while not initial.within(boundary_poly):
            initial_x = random.uniform(-300, -100)
            initial_y = random.uniform(-100, 100)
            initial = Point(initial_x, initial_y)
        initial = Motion_plan_state(initial_x, initial_y)

        res1 = testing.exploring(initial, habitats, 0.5, 5, 1, 50, True, 20.0, 500.0, weights=weight)
        print(res1["cost"])
        for i in range(len(res1["cost"][1])):
            cost_summary_ex[i].append(res1["cost"][1][i])

        res2 = testing.replanning(initial, habitats, 10.0, 100.0, 0.1)
        print(res2[2])
        for i in range(len(res2[2][1])):
            cost_summary_rp[i].append(res2[2][1][i])

    #calculate average cost for each term
    result1 = []
    for cost in cost_summary_ex:
        result1.append(statistics.mean(cost))
    result2 = []
    for cost in cost_summary_rp:
        result2.append(statistics.mean(cost))
    
    return [result2, result1]

def plot_summary_1(labels, summarys):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    weight1 = summarys[0]
    weight2 = summarys[1]
    weight3 = summarys[2]
    weight4 = summarys[3]

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5 * width, weight1, width, label='weight1')
    rects2 = ax.bar(x - 0.5 * width, weight2, width, label="weight2")
    rects3 = ax.bar(x + 0.5 * width, weight3, width, label='weight3')
    rects4 = ax.bar(x + 1.5 * width, weight4, width, label='weight4')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('average cost')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()

    plt.show()

def summary_2(start, goal, obstacle_array, boundary, habitats, shark_dict, sharkGrid, test_num, test_time, plot_interval, weights):
    '''generate the average cost of optimal paths of one weight scheme'''
    cost_list = [[]for _ in range(math.ceil(test_time//plot_interval))]
    improvement = []

    for _ in range(test_num):
        rrt = RRT(start, goal, boundary, obstacle_array, habitats)
        if weights[1] == "random time":
            plan_time = True
            if weights[2] == "trajectory time stamp":
                traj_time_stamp = True
            else:
                traj_time_stamp = False
        elif weights[1] == "random (x,y)":
            plan_time = False
            traj_time_stamp = False
        result = rrt.exploring(shark_dict, sharkGrid, plot_interval, 5, 1, 50,traj_time_stamp=traj_time_stamp, max_plan_time=test_time, plan_time=plan_time, weights=weights[0])
        if result:
            cost = result["cost list"]
            for i in range(len(cost)):
                cost_list[i].append(cost[i])
    
    cost_mean = []
    for i in range(len(cost_list)):
        temp_mean = statistics.mean(cost_list[i])
        if i >= 1:
            improvement.append("{:.0%}".format(temp_mean / cost_mean[-1]))
        cost_mean.append(temp_mean)
    
    #plot_summary_2(time_list, cost_list)
    #print(cost_mean, improvement)
    return cost_mean, improvement

def plot_summary_2(x_list, y_list):

    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('optimal sum cost')
    plt.title('RRT performance')

    plt.show()

def summary_3(start, goal, boundary, boundary_poly, obstacle_array, habitats, shark_dict, test_num, plan_time, plot_interval):
    '''draw average cost of optimal path from different weight schemes as a function of time'''
    results = []
    improvements = []
    time_list = [plot_interval + i * plot_interval for i in range(math.ceil(plan_time//plot_interval))]

    weight1 = [[1, -3, -1, -5], "random time", "trajectory time stamp"]
    weight2 = [[1, -3, -1, -5], "random time", "planning time stamp"]
    weight3 = [[1, -3, -1, -5], "random (x,y)"]
    weights = [weight1, weight2, weight3]

    cell_list = splitCell(boundary_poly, 10)
    sharkGrid = createSharkGrid('path_planning/AUVGrid_prob.csv', cell_list)
    
    for weight in weights:
        result, improvement = summary_2(start, goal, obstacle_array, boundary, habitats, shark_dict, sharkGrid, test_num, plan_time, plot_interval, weight)
        results.append(result)
        improvements.append(improvement)

    plt.figure(1)
    for i in range(len(results)):
        plt.plot(time_list, results[i], label=str(weights[i]))
    plt.ylabel('Optimal Path Cost')
    plt.xlabel('Planning Time')
    plt.title('Optimal Path Cost VS Planning Time')
    plt.legend()
    plt.show()
    plt.close()

    # plt.figure(2)
    # for i in range(len(improvements)):
    #     print(time_list[1:], improvements[i])
    #     plt.plot(time_list[1:], improvements[i], label=str(weights[i]))
    # plt.ylabel('Proportion Cost Optimization')
    # plt.xlabel('Planning Time')
    # plt.title('Percent Optimization over Planning Time')
    # plt.legend()
    # plt.show()
    # plt.close()

def plot_time_stamp(start, goal, boundary, obstacle_array, habitats):
    '''draw time stamp distribution of one rrt_rubins path planning algorithm'''
    rrt = RRT(start, goal, boundary, obstacle_array, habitats)
    result = rrt.exploring(habitats, 0.5, 5, 1, max_plan_time=10.0, weights=[1,-4.5,-4.5])
    time_stamp_list = result["time stamp"]
    bin_list = time_stamp_list.keys()
    num_time_list = []
    for time_bin in bin_list:
        num_time_list.append(len(time_stamp_list[time_bin]))
    
    plt.title("time stamp distribution")
    plt.xlabel("time stamp bin")
    plt.ylabel("number of motion_plan_states")
    #plt.xticks(self.bin_list)
    plt.bar(bin_list, num_time_list, color="g")
    
    plt.show()

#initialize start, goal, obstacle, boundary, habitats for path planning
start = catalina.create_cartesian(catalina.START, catalina.ORIGIN_BOUND)
start = Motion_plan_state(start[0], start[1])

goal = catalina.create_cartesian(catalina.GOAL, catalina.ORIGIN_BOUND)
goal = Motion_plan_state(goal[0], goal[1])

obstacles = []
for ob in catalina.OBSTACLES:
    pos = catalina.create_cartesian((ob.x, ob.y), catalina.ORIGIN_BOUND)
    obstacles.append(Motion_plan_state(pos[0], pos[1], size=ob.size))
for boat in catalina.BOATS:
    pos = catalina.create_cartesian((boat.x, boat.y), catalina.ORIGIN_BOUND)
    obstacles.append(Motion_plan_state(pos[0], pos[1], size=boat.size))
        
boundary = []
boundary_poly = []
for b in catalina.BOUNDARIES:
    pos = catalina.create_cartesian((b.x, b.y), catalina.ORIGIN_BOUND)
    boundary.append(Motion_plan_state(pos[0], pos[1]))
    boundary_poly.append((pos[0],pos[1]))
boundary_poly = Polygon(boundary_poly)
        
#testing data for habitats
habitats = []
for habitat in catalina.HABITATS:
    pos = catalina.create_cartesian((habitat.x, habitat.y), catalina.ORIGIN_BOUND)
    habitats.append(Motion_plan_state(pos[0], pos[1], size=habitat.size))
    
# testing data for shark trajectories
shark_dict1 = {1: [Motion_plan_state(-120 + (0.2 * i), -60 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    2: [Motion_plan_state(-65 - (0.2 * i), -50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    3: [Motion_plan_state(-110 + (0.2 * i), -40 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    4: [Motion_plan_state(-105 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    5: [Motion_plan_state(-120 + (0.2 * i), -50 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    6: [Motion_plan_state(-85 - (0.2 * i), -55 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    7: [Motion_plan_state(-270 + (0.2 * i), 50 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    8: [Motion_plan_state(-250 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)],
    9: [Motion_plan_state(-260 - (0.2 * i), 75 + (0.2 * i), traj_time_stamp=i) for i in range(1,501)], 
    10: [Motion_plan_state(-275 + (0.2 * i), 80 - (0.2 * i), traj_time_stamp=i) for i in range(1,501)]}

shark_dict2 = {1: [Motion_plan_state(-120 + (0.1 * i), -60 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-90 - (0.1 * i), -30 + (0.15 * i), traj_time_stamp=i) for i in range(302,501)], 
    2: [Motion_plan_state(-65 - (0.1 * i), -50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-95 + (0.15 * i), -20 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
    3: [Motion_plan_state(-110 + (0.1 * i), -40 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-80 + (0.15 * i), -70 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
    4: [Motion_plan_state(-105 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-135 + (0.12 * i), -25 + (0.07 * i), traj_time_stamp=i) for i in range(302,501)],
    5: [Motion_plan_state(-120 + (0.1 * i), -50 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-90 + (0.11 * i), -80 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
    6: [Motion_plan_state(-85 - (0.1 * i), -55 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-115 - (0.09 * i), -25 - (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
    7: [Motion_plan_state(-270 + (0.1 * i), 50 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-240 - (0.08 * i), 80 + (0.1 * i), traj_time_stamp=i) for i in range(302,501)], 
    8: [Motion_plan_state(-250 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-280 - (0.1 * i), 105 - (0.1 * i), traj_time_stamp=i) for i in range(302,501)],
    9: [Motion_plan_state(-260 - (0.1 * i), 75 + (0.1 * i), traj_time_stamp=i) for i in range(1,301)] + [Motion_plan_state(-290 + (0.08 * i), 105 + (0.07 * i), traj_time_stamp=i) for i in range(302,501)], 
    10: [Motion_plan_state(-275 + (0.1 * i), 80 - (0.1 * i), traj_time_stamp=i) for i in range(1,301)]+ [Motion_plan_state(-245 - (0.13 * i), 50 - (0.12 * i), traj_time_stamp=i) for i in range(302,501)]}
# sharkGrid1 = createSharkGrid('path_planning/AUVGrid_prob_500_straight.csv', splitCell(boundary_poly,10))
# sharkGrid2 = createSharkGrid('path_planning/AUVGrid_prob_500_turn.csv', splitCell(boundary_poly,10))

res = summary_1([1, -3, -1, -5], obstacles, boundary, habitats, shark_dict1, sharkGrid1, test_num=10)
plot_summary_1(["replaning", "one-time planning"], res)