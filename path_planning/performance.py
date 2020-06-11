import math
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np

from rrt_dubins import RRT
from motion_plan_state import Motion_plan_state

def summary_1(cost_funcs, num_habitats=10, test_num=100):
    '''
        generate a summary about each term of one specific cost function, given randomly chosen environment
    
        cost_func: a list of lists of weights assigned to each term in the cost function
        test_num: the number of tests to run under a specific cost function
        num_habitats: the number of randomly generated habitats

        output:
        cost_avr: a dictionary summarizing the result of each term of the cost function, 
            key will be weight i.e. w1, w2, ...
            value will be the average cost of each term
    '''
    
    start = Motion_plan_state(730, 280)
    goal = Motion_plan_state(900, 350)
    obstacle_array = [Motion_plan_state(757,243, size=10),Motion_plan_state(763,226, size=5)]
    boundary = [Motion_plan_state(700,200), Motion_plan_state(950,400)]
    testing = RRT(start, goal, obstacle_array, boundary)

    #inner list is the average cost of different environment for each weight 
    avr_list = [[] for _ in range(len(cost_funcs[0]))]
    #generate summary over different weights
    for cost_func in cost_funcs:

        count_list = []

        # a list of cost for each term of the optimal path for each environment
        cost_summary = [[] for _ in range(len(cost_func))]

        for i in range(test_num):

            count = 0
            
            #build random habitats
            habitats = []
            for _ in range(num_habitats):
                habitats.append(testing.get_random_mps(size_max=15))
            
            #find optimal path
            t_end = time.time() + 20.0
            cost_min = float("inf")
            cost_list = []  

            while time.time() < t_end:
                result = testing.exploring(habitats)
                count += 1
                if result is not None:
                    cost = result["cost"]
                    if cost[0] < cost_min:
                        cost_min = cost[0]
                        cost_list = cost[1]
            
            count_list.append(count)

            #append cost for each term to cost_summary list
            for i in range(len(cost_list)):
                cost_summary[i].append(cost_list[i])
        
        print(count_list)

        #calculate average cost for each term
        result = []
        for cost in cost_summary:
            result.append(statistics.mean(cost))
        
        for i in range(len(result)):
            avr_list[i].append(float("{:.3f}".format(result[i]/cost_func[i])))#normalize the result
    
    return avr_list

def plot_summary_1(labels, summary):
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    weight1 = summary[0]
    weight2 = summary[1]
    weight3 = summary[2]

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, weight1, width, label='weight1')
    rects2 = ax.bar(x, weight2, width, label="weight2")
    rects3 = ax.bar(x + width, weight3, width, label='weight3')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('average cost')
    ax.set_title('average cost in different weight schemes with randomly generated habitats')
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

    fig.tight_layout()

    plt.show()

def summary_2(start, goal, obstacle_array, boundary, habitats, test_num):
    cost_list = []
    time_list = []

    for _ in range(test_num):
        rrt = RRT(start, goal, obstacle_array, boundary)
        result = rrt.exploring(habitats, plan_time=False, test_time=5.0, weights=[1,-10,-10])
        time = result["time"]
        cost = result["cost list"]
        cost_list.append(cost)
        time_list.append(time)
    
    plot_summary_2(time_list, cost_list)
    return

def plot_summary_2(x_list, y_list):

    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], label="test case"+str(i+1))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('optimal sum cost')
    plt.title('RRT performance')

    plt.legend()
    plt.show()

start = Motion_plan_state(10,15)
goal = Motion_plan_state(7,4)
boundary = [Motion_plan_state(0,0), Motion_plan_state(100,100)]
obstacle_array = [Motion_plan_state(5,7, size=5),Motion_plan_state(14,22, size=3)]
habitats = [Motion_plan_state(63,23, size=5), Motion_plan_state(12,45,size=7), Motion_plan_state(51,36,size=5), Motion_plan_state(45,82,size=5),\
    Motion_plan_state(60,65,size=10), Motion_plan_state(80,79,size=5),Motion_plan_state(85,25,size=6)]
summary_2(start, goal, obstacle_array, boundary, habitats, 10)