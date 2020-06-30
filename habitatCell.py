"""
A wrapper class to represent state of an habitat cell
    Many habitat cells will make up a habitat area
    including x, y, z, size, and the number of time visited
"""
class HabitatCell:
    def __init__(self, x, y, habitat_id, side_length = 1):
        self.x = x
        self.y = y
        self.side_length = side_length
        self.habitat_id = habitat_id

    def __repr__(self):
        return "Habitat: [id=" + str(self.habitat_id) + ", x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" +\
            str(self.side_length) + "]"

    def __str__(self):
        return "Habitat: [id=" + str(self.habitat_id) + ", x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" +\
            str(self.side_length) + "]"
