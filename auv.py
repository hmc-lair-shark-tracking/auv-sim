from motion_plan_state import Motion_plan_state
import math
import numpy as np
import constants as const
import random
from sharkState import SharkState
def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians
    """
    if -math.pi <= ang <= math.pi:
        return ang
    elif ang > math.pi: 
        ang += (-2 * math.pi)
        return angle_wrap(ang)
    elif ang < -math.pi: 
        ang += (2 * math.pi)
        return angle_wrap(ang)

class Auv:
    def __init__(self, x, y, z, theta, velocity_1,w_1, curr_traj_pt_index,i):
        self.state = Motion_plan_state( x, y, z = 0, theta = 0)
        self.velocity_1 = velocity_1
        self.w_1 = w_1
        self.curr_traj_pt_index = 0
        self.i = i 
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.x_list += [self.state.x]
        self.y_list += [self.state.y]
        self.z_list += [self.state.z]
        self.sensor_time = const.NUM_ITER_FOR_NEW_SENSOR_DATA
        self.shark_sensor_data_list = []

    def send_trajectory_to_actuators(self):
        # TODO: For now this should just update AUV States?
        const.SIM_TIME_INTERVAL
        self.calculate_new_auv_state(const.SIM_TIME_INTERVAL)

    def calculate_new_auv_state(self, delta_t):
        """ 
        Calculate new x, y and theta

        Parameters: 
            v - linear velocity of the robot (m/s)
            w - angular veloctiy of the robot (rad/sg)
            delta_t - time step (sec)
        """
        # returns random velocity and w_velocity
        self.state.x = self.state.x + self.velocity_1 * math.cos(self.state.theta)*delta_t
        self.state.y = self.state.y + self.velocity_1 * math.sin(self.state.theta)*delta_t
        self.state.theta = angle_wrap(self.state.theta + self.w_1 * delta_t)
    
        self.x_list += [self.state.x]
        self.y_list += [self.state.y]
        self.z_list += [self.state.z]
        
    def get_auv_sensor_measurements(self, curr_time):
        """
            Return an Motion_plan_state object that represents the measurements
                of the auv's x,y,z,theta position with a time stamp
            The measurement has random gaussian noise
            
            # 0 is the mean of the normal distribution you are choosing from
            # 1 is the standard deviation of the normal distribution

            # np.random.normal returns a single sample drawn from the parameterized normal distribution
            # we actually omitted the third parameter which determines the number of samples that we would like to draw
        """
        x = self.state.x + np.random.normal(0,1)
        y = self.state.y + np.random.normal(0,1)
        z = self.state.z + np.random.normal(0,1)
        theta = angle_wrap(self.state.theta + np.random.normal(0,1))
        
        return Motion_plan_state(x, y, z, theta, curr_time)

    def track_trajectory(self, trajectory, new_trajectory, curr_time):
        """
        Return an Motion_plan_state object representing the trajectory point TRAJ_LOOK_AHEAD_TIME sec ahead
        of current time

        Parameters: 
            trajectory - a list of trajectory points, where each element is 
            a Motion_plan_state object that consist of time stamp, x, y, z,theta
        """
        # only increment the index if it hasn't reached the end of the trajectory list
        if new_trajectory == True:
            self.curr_traj_pt_index = 0
        while (self.curr_traj_pt_index < len(trajectory)-1) and\
            (curr_time + const.TRAJ_LOOK_AHEAD_TIME) > trajectory[self.curr_traj_pt_index].traj_time_stamp: 
            self.curr_traj_pt_index += 1

        return trajectory[self.curr_traj_pt_index]

    def track_way_point(self, way_point):
        """
        Calculates the v&w to get to the next point along the trajectory

        way_point - a motion_plan_state object, represent the trajectory point that we are tracking
        """
        # K_P and v are stand in values
        K_P = 0.5
        # v = 12
         # TODO: currently change it to a very unrealistic value to show the final plot faster
       
        angle_to_traj_point = math.atan2(self.state.x - self.state.y, way_point.x - self.state.x) 
        self.w_1 = K_P * angle_wrap(angle_to_traj_point - self.state.theta) #proportional control
        
        return self.velocity_1, self.w_1
    


    def get_all_sharks_sensor_measurements(self, shark_state_dict, auv_sensor_data):
        """
        Modify the data member self.shark_state_dict if there is new sensor data
            key = id of the shark & value = the shark's range and bearing (stored as a sharkState object)

        Parameter: 
            shark_state_dict - a dictionary, containing the shark's states at a given time
            auv_sensor_data - a motion_plan_state object, containting the auv's position

        Return Value:
            True - if it has been 2 sec and there are new shark sensor measurements
            False - if it hasn't been 2 sec
        """
        # decide to sensor_time an integer because floating point addition is not as reliable
        # each iteration through the main navigation loop is 0.1 sec, so 
        #   we need 20 iterations to return a new set of sensor data
        if self.sensor_time == const.NUM_ITER_FOR_NEW_SENSOR_DATA:
            self.shark_sensor_data_list = []
            #print(" len of shark_state_dict", len(shark_state_dict))
            # iterate through all the sharks that we are tracking
            for shark_id in shark_state_dict: 
                shark_data = shark_state_dict[shark_id]
                delta_x = shark_data.x - self.state.x
                delta_y = shark_data.y - self.state.y

                range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
                
                bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5
    
                Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
                Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) - self.state.theta + bearing_random)
                # updates new x, y here 
                #print("new shark ",  shark_id, "shark coordinates", shark_data.x, shark_data.y)
                self.shark_sensor_data_list.append([self.state.x, self.state.y, self.state.theta, Z_shark_range, Z_shark_bearing,  shark_id, shark_data.x, shark_data.y])

            
            # reset the 2 sec time counter
            self.sensor_time = 0
            print("================== has new data")
            return self.shark_sensor_data_list
        else: 
            self.sensor_time += 1

            return self.shark_sensor_data_list