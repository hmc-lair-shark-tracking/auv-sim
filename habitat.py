"""
A wrapper class to represent state of a habitat, which consists of many habitat cells
    including x, y, z, size, and the number of time visited
"""
class Habitat:
    def __init__(self, x, y, id_in, side_length = 1, num_of_time_visited = 0):
        self.x = x
        self.y = y
        self.id = id_in
        self.side_length = side_length
        self.num_of_time_visited = num_of_time_visited

    def __repr__(self):
        return "Habitat: [x=" + str(self.x) + ", y="  + str(self.y) + ", id=" + str(self.id) + ", side length=" +\
            str(self.side_length) + ", visited=" + str(self.num_of_time_visited) + "]"

    def __str__(self):
        return "Habitat: [x=" + str(self.x) + ", y="  + str(self.y) + ", id=" + str(self.id) + ", side length=" +\
            str(self.side_length) + ", visited=" + str(self.num_of_time_visited) + "]"