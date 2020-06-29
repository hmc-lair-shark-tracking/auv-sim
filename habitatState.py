"""
A wrapper class to represent state of an habitat
    including x, y, z, size, and the number of time visited
"""
class HabitatState:
    def __init__(self, x, y, habitat_id, side_length = 1, num_of_time_visited = 0):
        self.x = x
        self.y = y
        self.side_length = side_length
        self.habitat_id = habitat_id
        self.num_of_time_visited = 0

    def __repr__(self):
        return "Habitat: [id=" + str(self.habitat_id) + ", x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" +\
            str(self.side_length)  + ", visited=" +  str(self.num_of_time_visited) + "]"

    def __str__(self):
        return "Habitat: [id=" + str(self.habitat_id) + ", x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" +\
            str(self.side_length)  + ", visited=" +  str(self.num_of_time_visited) + "]"
