import matplotlib.pyplot as plt
import math

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

    
    def particle_cluster_over_time(self, list_of_news, list_of_number_of_particles):
            plt.plot(list_of_news, color = '#eb5282')
            plt.ylabel('record time (s)')
            plt.xlabel('number of particles')









    