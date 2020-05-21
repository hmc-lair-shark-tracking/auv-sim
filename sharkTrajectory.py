from motion_plan_state import Motion_plan_state

"""
A class to store the trajectory of a given shark
"""
class SharkTrajectory:
    def __init__(self, shark_id, x_pos_array, y_pos_array, x_vel_array = [], y_vel_array = []):
        """
        Note:
            x_pos_array and y_pos_array are passed in as array of strings when they get read from 
            the csv file
        """
        self.id = shark_id
        self.traj_pts_array = []
        # keeps track of the index into shark's trajectory array 
        self.index = 0

        # decided to update the position arrays as we draw the shark's position
        # use these arrays to draw the shark's trajectory & for summary plots at the
        # end of the simulation
        self.x_pos_array = []
        self.y_pos_array = []
        self.z_pos_array = []
        
        # use map to convert the array of strings to array of float
        self.x_vel_array = list(map(lambda x: float(x), x_vel_array))
        self.y_vel_array = list(map(lambda y: float(y), y_vel_array))

        for i in range(len(x_pos_array)):
            # the time_stamp will be spaced as 2 sec based on what we discussed in the meeting
            self.traj_pts_array.append(\
                Motion_plan_state(x = float(x_pos_array[i]), y = float(y_pos_array[i]), time_stamp = i * 2))

    def store_positions(self, x, y, z): 
        """
        Helper function to store new position into the array
        """
        self.x_pos_array.append(x)
        self.y_pos_array.append(y)
        self.z_pos_array.append(z)

    def __repr__(self):
        return "shark trajectory #" + str(self.id) + " with " + str(len(self.traj_pts_array)) + " trajectory pts"

    def __str__(self):
        return "shark trajectory #" + str(self.id) + " with " + str(len(self.traj_pts_array)) + " trajectory pts"
