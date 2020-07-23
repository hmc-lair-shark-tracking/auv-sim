import numpy as np

from habitatCell import HabitatCell
from habitat import Habitat

"""
A wrapper class to discretize the environment into grids

Maybe to update the number of time visited, we have an extra array storing that in that class. Habitat State will be store the index into 
that arrray instead. Then, we can increment the number of time that way. In this sense, we will be able to have different shape in the future?
"""

class HabitatGrid:
    def __init__(self, env_x, env_y, env_size_x, env_size_y, habitat_side_length = 10, cell_side_length = 1):
        self.env_x = env_x
        self.env_y = env_y
        self.env_size_x = env_size_x
        self.env_size_y = env_size_y
        self.habitat_side_length = habitat_side_length
        self.cell_side_length = cell_side_length
        
        row_hab_id = 0
        col_hab_id = 0

        self.habitat_cell_grid = []
        
        self.habitat_array = []

        for row in range(int(self.env_size_y) // int(self.cell_side_length)):
            self.habitat_cell_grid.append([])
            if row % (int(self.habitat_side_length) // int(self.cell_side_length)) == 0 and row != 0:
                # inside a new habitat
                row_hab_id = col_hab_id + 1

            col_hab_id = row_hab_id
            
            for col in range(int(self.env_size_x) // int(self.cell_side_length)):
                hab_x = env_x + col * cell_side_length
                hab_y = env_y + row * cell_side_length

                if col % (int(self.habitat_side_length) // int(self.cell_side_length))== 0:
                    if col != 0:
                        # inside a new habitat
                        col_hab_id += 1
                    if row % (int(self.habitat_side_length) // int(self.cell_side_length)) == 0:
                        # need to add a new habitat into the habitat array
                        self.habitat_array.append(Habitat(hab_x, hab_y, col_hab_id, self.habitat_side_length))

                self.habitat_cell_grid[row].append(HabitatCell(hab_x, hab_y, col_hab_id, side_length = self.cell_side_length))


    def print_habitat_cell_grid(self):
        for row in self.habitat_cell_grid:
            for habitat_cell in row:
                print(habitat_cell.habitat_id, end=' ')
            print()

    
    def within_habitat_env(self, auv_pos):
        auv_x = auv_pos[0]
        auv_y = auv_pos[1]

        return ((auv_x >= self.env_x) and (auv_x < self.env_x + self.env_size_x))\
            and ((auv_y >= self.env_y) and (auv_y < self.env_y + self.env_size_y))

    
    def inside_habitat(self, auv_pos):
        """

        Warning:
            - Assume that auv_pos is only positive
        """
        auv_x = auv_pos[0]
        auv_y = auv_pos[1]

        hab_index_row = int(auv_y / self.cell_side_length)
        hab_index_col = int(auv_x / self.cell_side_length)

        if hab_index_row >= len(self.habitat_cell_grid):
            print("auv is out of the habitat environment bound verticaly")
            return False
        
        if hab_index_col >= len(self.habitat_cell_grid[0]):
            print("auv is out of the habitat environment bound horizontally")
            return False

        return self.habitat_cell_grid[hab_index_row][hab_index_col]


    def distance_from_grid_boundary(self, auv_pos):
        """

        Return:
            an array represent the distance from the four walls: [top wall, right wall, bottom wall, left wall]
        """
        auv_x = auv_pos[0]
        auv_y = auv_pos[1]

        top_wall_y = self.env_y + self.env_size_y
        right_wall_x = self.env_x + self.env_size_x
        left_wall_x = self.env_x
        bottom_wall_y = self.env_y

        return np.array([top_wall_y - auv_y, right_wall_x - auv_x, auv_y - bottom_wall_y, auv_x - left_wall_x])