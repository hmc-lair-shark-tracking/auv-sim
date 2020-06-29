from habitatState import HabitatState

"""
A wrapper class to discretize the environment into grids
"""

class HabitatGrid:
    def __init__(self, env_x, env_y, env_size_x, env_size_y, cell_side_length = 1, habitat_side_length = 10):
        self.env_x = env_x
        self.env_y = env_y
        self.env_size_x = env_size_x
        self.env_size_y = env_size_y
        self.cell_side_length = cell_side_length
        self.habitat_side_length = habitat_side_length
        
        row_hab_id = 0
        col_hab_id = 0

        for row in range(self.env_size_y // self.cell_side_length):
            self.habitat_grid.append([])
            if row % self.habitat_side_length == 0 and row != 0:
                row_hab_id = col_hab_id + 1

            col_hab_id = row_hab_id
            
            for col in range(self.env_size_x // self.cell_side_length):
                if col % self.habitat_side_length == 0 and col != 0:
                    # inside a new habitat
                    col_hab_id += 1
                hab_x = env_x + col * cell_side_length
                hab_y = env_y + row * cell_side_length
                self.habitat_grid[row].append(HabitatState(hab_x, hab_y, col_hab_id))


    def print_habitat_grid(self):
        for row in self.habitat_grid:
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

        hab_index_row = int(auv_y)
        hab_index_col = int(auv_x)

        if hab_index_row >= len(self.habitat_grid):
            print("auv is out of the habitat environment bound verticaly")
            return False
        
        if hab_index_col >= len(self.habitat_grid[0]):
            print("auv is out of the habitat environment bound horizontally")
            return False

        return self.habitat_grid[hab_index_row][hab_index_col]


def main():
    testing_grid = HabitatGrid(0, 0, 50, 50)

    testing_grid.print_habitat_grid()

                
