"""
A wrapper class to represent state of an habitat
    including x, y, z, size, and the number of time visited
"""
class HabitatState:
    def __init__(self, x, y, z, size, num_of_time_visited = 0):
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.num_of_time_visited = 0

    def __repr__(self):
        return "Habitat: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + ", size=" +\
            str(self.size)  + ", visited=" +  str(self.num_of_time_visited) + "]"

    def __str__(self):
        return "Habitat: [x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + ", size=" +\
            str(self.size)  + ", visited=" +  str(self.num_of_time_visited) + "]"
