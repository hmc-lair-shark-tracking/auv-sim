import matplotlib.pyplot as plt
import math
import numpy as np


class Figure:
    def plot_distance(self, new_mean_x, new_mean_y, new_coordinate_x, new_coordinate_y):
        index_1 = -1
        index_2 = -1
        index_3 = -1
        list_x = []
        list_y = []
        dist = []
        for coordinate in new_mean_x:
            index_1+= 1
            x = coordinate - new_coordinate_x[index_1]
            x = x**2 
            list_x.append(x)
        for cordinate in new_mean_y:
            index_2+= 1
            y = coordinate - new_coordinate_y[index_2]
            y = y**2 
            list_y.append(y)
        for value in list_x:
            index_3 += 1
            d = math.sqrt(value + list_y[index_3] )
            dist.append(d)
        plt.plot(dist)
        plt.ylabel('distance')
        plt.xlabel('time')
    
    def coordinate_plotter(self, x_mean_over_time, y_mean_over_time, final_new_shark_x, final_new_shark_y, sim_time_list):
        # plots (x,y) of particles in  blue and plots (x, y) of shark in red
        new_sim_time_list = []
        new_list_x = []
        new_list_y = []
        index_1 = -1
        index_2 = -1
        index_3 = -1
        for coordinate in final_new_shark_y:
            index_1 += 1
            solution = abs(coordinate - y_mean_over_time[index_1])
            new_list_y.append(solution)
        for coordinate in final_new_shark_x:
            index_2 += 1
            solution = abs(coordinate - x_mean_over_time[index_2])
            new_list_x.append(solution)
        for value in new_list_x:
            index_3 += 1
            if value == new_list_y[index_3]:
                print(sim_time_list[index_3])
                new_sim_time_list.append(sim_time_list[index_3])
        plt.plot(new_list_y, color = '#eb5282')
        plt.plot(new_list_x, color = '#5053f2')
        plt.ylabel('difference between mean and shark x- blue, y- red')
        plt.xlabel('time')
        return new_sim_time_list
    
    
    def max_plotter(self, max_x, max_y, new_coordinate_x, new_coordinate_y):
        #plots max particle weights' coordinates
        index_1 = -1
        index_2 = -1
        index_3 = -1
        new_list_y = []
        new_list_x = []
        for coordinate in new_coordinate_y:
            index_1 += 1
            solution = abs(coordinate - max_y[index_1])
            new_list_y.append(solution)
        for coordinate in new_coordinate_x:
            index_2 += 1
            solution = abs(coordinate - max_x[index_2])
            new_list_x.append(solution)

        
        plt.plot(new_list_y, color = '#eb5282')
        plt.plot(new_list_x, color = '#5053f2')
        plt.ylabel('difference between mean and shark x- blue, y - red')
        plt.xlabel('time')

    def range_plotter(self, x_mean_over_time, y_mean_over_time, final_new_shark_x, final_new_shark_y, sim_time_list):
        #plots max particle weights' coordinates
        index_1 = -1
        index_3 = -1
        mean_error = 0
        range_list = []
        final_time_list = []
        mean_list = []
        for coordinate in y_mean_over_time:
            index_1 += 1
            solution = abs(coordinate - final_new_shark_y[index_1])**2 
            solution2 = abs(x_mean_over_time[index_1]- final_new_shark_x[index_1])**2
            solution3 = math.sqrt(solution + solution2)
            range_list.append(solution3)
        return range_list


    def range_list_function(self, range_list, sim_time_list):
        index = -1
        mean_error = 0
        sum_1 = 0
        final_time_list = []
        mean_list = []
        sum_1 +=  range_list[-5]
        sum_1 += range_list[-4]
        sum_1 += range_list[-3]
        sum_1 += range_list[-2]
        sum_1 += range_list[-1]
        mean_error = (sum_1 / 5)
        for value in range_list:
            index += 1
            mean_list.append(mean_error)
            if value <= (1.1) * mean_error:
                final_time_list.append(sim_time_list[index])
        print("mean error")
        print(mean_error)
        print("time list")
        print(final_time_list)

        plt.plot(range_list, color = '#eb5282')
        plt.plot(mean_list, color = '#038cfc')
        plt.ylabel('range error (meters) ')
        plt.xlabel('time (s)')

    """
    # plots just to plot time and convergence, not really useful
    def mean_convergence_plot(self):
        list_of_2_auv = [4.727, 5.5, 5.772727, 4.8636]
        plt.plot(list_of_2_auv, color =  '#038cfc' )
        plt.ylabel('Mean Steady State Error (meters)')
        plt.xlabel('Number of AUVs')

    def time_convergence_plot(self):

        list_of_1_auv = [18.421, 16.31, 16.611, 14.1875]
        
        plt.plot(list_of_1_auv , color = '#eb5282')
        
        plt.ylabel('time (s)')
        plt.xlabel('NUM OF AUV')
    """
    def mean_over_time(self, final_range_error_list):
        # generates average range error list for each number of auv over x number of trials
        final_mean_list = []
        for index in range(len(final_range_error_list[0])):
            sum = 0
            for error in final_range_error_list:
                sum += error[index]
            individidual_mean = sum/ len(final_range_error_list)
            final_mean_list.append(individidual_mean)

        print("final range error for each time step")
        print(final_mean_list)
        
        plt.plot(final_mean_list , color = '#eb5282')
        plt.ylabel('range error (m)')
        plt.xlabel('Time (s)')
    def combined_plotter(self, first_auv, second_auv, third_auv, fourth_auv):
        # combined plot for all auv range error lists
        plt.plot(first_auv , '--', label= '1 AUV', color = '#718bbf')
        plt.plot(second_auv ,':', label = '2 AUVS', color = '#35373b')
        plt.plot(third_auv, '-.', label = '3 AUVS', color = '#4d7bd6' )
        plt.plot(fourth_auv, '-' , label = '4 AUVS', color = '#0554f0' )
        plt.legend()
        plt.ylabel('Range Error (m)')
        plt.xlabel('Time (s)')












    