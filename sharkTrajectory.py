from objectState import ObjectState

"""
A class to store the trajectory of a given shark
"""
class sharkTrajectory:
    def __init__(self, shark_id, x_pos_array, y_pos_array):
        self.id = shark_id
        self.trajectory_array = []

        for i in range(len(x_pos_array)):
            self.trajectory_array.append(ObjectState(x_pos_array[i], y_pos_array[i]))

    def __repr__(self):
        return "shark trajectory #" + str(self.id) + "\n" + self.trajectory_array

    def __str__(self):
        return "shark trajectory #" + str(self.id) + "\n" + self.trajectory_array