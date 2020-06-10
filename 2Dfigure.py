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
            x = oordinate - new_coordinate_x[index_1]
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

        self.plt.plot(dist)



    