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
from cost import habitat_shark_cost_func, habitat_shark_cost_point

def summary_1(weight, obstacles, boundary, habitats, shark_dict, sharkGrid, cell_list, test_num=10):
    '''
        generate a summary about each term of one specific cost function, given randomly chosen environment
    
        test_num: the number of tests to run under a specific cost function

        output:
        cost_avr: a dictionary summarizing the result of each term of the cost function, 
            key will be weight i.e. w1, w2, ...
            value will be the average cost of each term
    '''
    # sharkOccupancyGrid = SharkOccupancyGrid(10, boundary, 50, 50, cell_list)
    for comp in [4, 5, 7, 8, 9, 10]:
        cost_summary_ex_whabitat_cal = []
        cost_summary_ex_whabitat_plan = []
        cost_summary_ex_wohabitat = []
        # cost_summary_rp = []
        
        # ran_hab = int(random.uniform(0, comp))
        temp_habitats = []
        # temp_shark = shark_dict

        while len(temp_habitats) < comp and len(temp_habitats) < len(habitats):
            temp = math.floor(random.uniform(0, len(habitats)))
            while temp >= len(habitats):
                temp = math.floor(random.uniform(0, len(habitats)))
            temp_habitats.append(habitats[temp])
        # ran_shark = comp - len(temp_habitats)
        # while len(list(temp_shark.keys())) < ran_shark and len(list(temp_shark.keys())) < len(list(shark_dict.keys())):
        #     temp = math.floor(random.uniform(1, len(list(shark_dict.keys()))))
        #     while temp >= len(list(shark_dict.keys())):
        #         temp = math.floor(random.uniform(0, len(habitats)))
        #     temp_shark[temp] = shark_dict[temp]
        print(len(temp_habitats))
        temp_sharkGrid = sharkGrid

        temp_cost_ex_whabitat_cal = []
        temp_cost_ex_wohabitat = []
        temp_cost_ex_whabitat_plan = []
        # temp_cost_rp = []
        for _ in range(test_num):
            initial_x = random.uniform(-300, -100)
            initial_y = random.uniform(-100, 100)
            initial = Point(initial_x, initial_y)
            while not initial.within(boundary_poly):
                initial_x = random.uniform(-300, -100)
                initial_y = random.uniform(-100, 100)
                initial = Point(initial_x, initial_y)
            initial = Motion_plan_state(initial_x, initial_y)

            testing = RRT(boundary, obstacles, temp_sharkGrid, cell_list)
            res1 = testing.exploring(initial, [], 0.5, 5, 1, 50, True, 20.0, 500.0, weights=weight)
            path = res1['path'][0]
            cost = habitat_shark_cost_func(path, path[-1].traj_time_stamp, temp_habitats, temp_sharkGrid, weight)
            # print(res1["cost"], cost)
            temp_cost_ex_wohabitat.append(res1["cost"][0])
            temp_cost_ex_whabitat_cal.append(cost[0])

            res3 = testing.exploring(initial, temp_habitats, 0.5, 5, 1, 50, True, 20.0, 500.0, weights=weight)
            # print(res3["cost"])
            temp_cost_ex_whabitat_plan.append(res3["cost"][0])

            # res2 = testing.replanning(initial, habitats, 10.0, 100.0, 0.1, weight)
            # print(res2[2])
            # temp_cost_rp.append(res2[2][0])

        cost_summary_ex_whabitat_cal.append(statistics.mean(temp_cost_ex_whabitat_cal))
        cost_summary_ex_whabitat_plan.append(statistics.mean(temp_cost_ex_whabitat_plan))
        cost_summary_ex_wohabitat.append(statistics.mean(temp_cost_ex_wohabitat))
        # cost_summary_rp.append(statistics.mean(temp_cost_rp))
        print(str(comp) + "consider habitats in planning: " + str(cost_summary_ex_whabitat_plan) + 
            ", not consider habitats: "+ str(cost_summary_ex_wohabitat) + 
            ", consider habitats in calculation: " + str(cost_summary_ex_whabitat_cal))
    
    return 

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
    plt.figure(1)

    for planner, cost in y_list.items():
        plt.plot(x_list, cost, label=planner)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.legend()
    plt.xticks(x_list)
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

def summary_4(rrt, habitats, shark_dict, weight):
    # res = rrt.exploring(Motion_plan_state(-200, 0), habitats, 0.5, 5, 2, 50, traj_time_stamp=True, max_plan_time=10, max_traj_time=500, plan_time=True, weights=weight)
    # traj = res["path"][0]
    # print(traj, res['cost'])

    res = rrt.replanning(Motion_plan_state(-200, 0), habitats, 10.0, 500.0, 0.1)
    traj = res[0]
    # print(traj, res[2])

    #initialize
    total_cost = 0
    # num_steps = 0
    t = 1
    visited = [False for _ in range(len(habitats))]

    #helper
    curr = 0
    time_diff = float("inf")
    bin_list = list(shark_dict.keys())
    curr_time = 0
    
    while t <= (traj[-1].traj_time_stamp//1):
        diff = abs(traj[curr].traj_time_stamp - t)
        while diff <= time_diff:
            time_diff = diff
            curr += 1
            if curr == len(traj) -1:
                break
            diff = abs(traj[curr].traj_time_stamp - t)
        curr = curr - 1
        time_diff = float("inf")
        if traj[curr].traj_time_stamp > bin_list[curr_time][1]:
            curr_time += 1
        AUVGrid = shark_dict[bin_list[curr_time]]

        cost, visited = habitat_shark_cost_point(traj[curr], habitats, visited, AUVGrid, weight)
        total_cost += cost
        t += 1
        # num_steps += 1
    
    return total_cost / traj[-1].traj_time_stamp

#initialize start, goal, obstacle, boundary, habitats for path planning
# start = catalina.create_cartesian(catalina.START, catalina.ORIGIN_BOUND)
# start = Motion_plan_state(start[0], start[1])

# goal = catalina.create_cartesian(catalina.GOAL, catalina.ORIGIN_BOUND)
# goal = Motion_plan_state(goal[0], goal[1])

# convert to environment in casrtesian coordinates 
environ = catalina.create_environs(catalina.OBSTACLES, catalina.BOUNDARIES, catalina.BOATS, catalina.HABITATS)

obstacles = environ[0] + environ[2]
boundary = environ[1]
habitats = environ[3]
boundary_poly = [(mps.x, mps.y) for mps in boundary]
boundary_poly = Polygon(boundary_poly)

# cell_list = splitCell(boundary_poly,10)
    
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
# sharkGrid1 = createSharkGrid('path_planning/AUVGrid_prob_500_straight.csv', cell_list)
# sharkGrid2 = createSharkGrid('path_planning/shark_data/AUVGrid_prob_500_turn.csv', cell_list)