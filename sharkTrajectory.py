from objectState import ObjectState

"""
A class to store the trajectory of a given shark
"""
class SharkTrajectory:
    def __init__(self, shark_id, x_pos_array, y_pos_array):
        self.id = shark_id
        self.traj_pts_array = []
        self.x_pos_array = []
        self.y_pos_array = []
        self.z_pos_array = []

        for i in range(len(x_pos_array)):
            self.traj_pts_array.append(ObjectState(float(x_pos_array[i]), float(y_pos_array[i])))

    def store_positions(self, x, y, z): 
        self.x_pos_array.append(x)
        self.y_pos_array.append(y)
        self.z_pos_array.append(z)

    def __repr__(self):
        return "shark trajectory #" + str(self.id) + " with " + str(len(self.traj_pts_array)) + " trajectory pts"

    def __str__(self):
        return "shark trajectory #" + str(self.id) + " with " + str(len(self.traj_pts_array)) + " trajectory pts"
