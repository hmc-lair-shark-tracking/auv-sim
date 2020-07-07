"""
A wrapper class to represent state of a grid cell in RRT planning
"""
class Grid_cell_RRT:
    def __init__(self, x, y, side_length = 1):
        """
        Parameters:
            x - x coordinates of the bottom left corner of the grid cell
            y - y coordinates of the bottom left corner of the grid cell
        """
        self.x = x
        self.y = y
        self.side_length = side_length
        self.node_list = []

    def has_node(self):
        """
        Return:
            True - if there is any RRT nodes in the the grid cell
        """
        return (not (self.node_list == []))

    def __repr__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.node_list)

    def __str__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.node_list)