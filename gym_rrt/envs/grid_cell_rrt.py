import numpy as np


"""
============================================================================

    Helper Functions

============================================================================
"""
def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians

    Note: 
        Because Python does not encourage importing files from the parent module, we have to place this angle wrap here. If we don't want to do this, we can possibly organize this so auv_env is in the parent folder?
    """
    if -np.pi <= ang <= np.pi:
        return ang
    elif ang > np.pi: 
        ang += (-2 * np.pi)
        return angle_wrap(ang)
    elif ang < -np.pi: 
        ang += (2 * np.pi)
        return angle_wrap(ang)


"""
A wrapper class to represent state of a grid cell in RRT planning
"""
class Grid_cell_RRT:
    def __init__(self, x, y, hasObstacle, side_length = 1, num_of_subsections = 8):
        """
        Parameters:
            x - x coordinates of the bottom left corner of the grid cell
            y - y coordinates of the bottom left corner of the grid cell
            side_length - side_length of the square grid cell
            sub_sections - how many subsections (based on theta) should the grid cell be split into
                the subsection are created in counter-clock wise direction (similar to a unit circle)
                However, once theta > pi, it becomes negative
        """
        self.x = x
        self.y = y
        self.side_length = side_length
        self.subsection_cells = []

        self.hasObstacle = hasObstacle

        self.delta_theta = float(2.0 * np.pi) / float(num_of_subsections)
        
        theta = 0.0
        # the node list will go in counter-clock wise direction
        for i in range(num_of_subsections):
            self.subsection_cells.append(self.Subsection_grid_cell_RRT(theta, self.hasObstacle))
            theta = angle_wrap(theta + self.delta_theta)

    def has_node(self):
        """
        Return:
            True - if there is any RRT nodes in the the grid cell
        """
        for subsection in self.subsection_cells:
            if subsection.node_array != []:
                return True
        return False

    def __repr__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.subsection_cells)

    def __str__(self):
        return "RRT Grid: [x=" + str(self.x) + ", y="  + str(self.y) + ", side length=" + str(self.side_length) +\
             "], node list: " + str(self.subsection_cells)


    class Subsection_grid_cell_RRT:
        def __init__(self, theta, hasObstacle):
            self.theta = theta
            self.hasObstacle = hasObstacle
            self.node_array = []

        def __repr__(self):
            return "Subsec: theta=" + str(self.theta) + ", hasObstacle?: " + str(self.hasObstacle) + ", node list: " + str(self.node_array)

        def __str__(self):
            return "Subsec: theta=" + str(self.theta) + ", hasObstacle?: " + str(self.hasObstacle) + ", node list: " + str(self.node_array)