import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import pandas as pd
import csv
# import 3 data representation class
from objectState import ObjectState
from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
from motion_plan_state import Motion_plan_state


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



class RobotSim:
    def __init__(self, init_x, init_y, init_z, init_theta):
        # initialize auv's data
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.theta = init_theta

        # need lists to keep track of all the points if we want to
        #   connect lines between the position points and create trajectories
        self.x_list = [init_x]
        self.y_list = [init_y]
        self.z_list = [init_z]

        self.shark_x_list = []
        self.shark_y_list = []
        self.shark_z_list = []

        # initialize the 3d scatter position plot for the auv and shark
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # keep track of the current time that we are in
        # each iteration in the while loop will be assumed as 0.1 sec
        self.curr_time = 0

        # index for which trajectory point that we should
        # keep track of
        self.curr_traj_pt_index = 0

        # create a square trajectory (list of objectState)
        # with parameter: v = 1.0 m/s and delta_t = 0.5 sec
        self.testing_trajectory = self.get_auv_trajectory(1, 0.5)

        self.shark_testing_trajectories = []


    def get_auv_state(self):
        """
        Return a Motion_plan_state representing the orientation and the time stamp
        of the robot
        """
        return Motion_plan_state(self.x, self.y, theta = self.theta, time_stamp=self.curr_time)


    def get_shark_state(self):
        """
        Return a Motion_plate_state representing the orientation
        of the shark (goal)
        """
        shark_X = 10
        shark_Y = 10
        shark_Z = -15
        shark_Theta = 0

        self.shark_x_list += [shark_X]
        self.shark_y_list += [shark_Y]
        self.shark_z_list += [shark_Z]

        return Motion_plan_state(shark_X, shark_Y, theta = shark_Theta)


    def get_auv_sensor_measurements(self):
        """
        Return an ObjectState object that represents the measurements
            of the auv's x,y,z,theta position with a time stamp
        The measurement has random gaussian noise
        """
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution

        # np.random.normal returns a single sample drawn from the parameterized normal distribution
        # we actually omitted the third parameter which determines the number of samples that we would like to draw

        return ObjectState(self.x + np.random.normal(0,1), self.y + np.random.normal(0,1), self.z + np.random.normal(0,1),\
            angle_wrap(self.theta + np.random.normal(0,1)), self.curr_time)


    def get_shark_sensor_measurements(self, currSharkX, currSharkY, currAuvX, currAuvY):
        delta_x = currSharkX - currAuvX
        delta_y = currSharkY - currAuvY
        range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
        bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

        Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
        Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) + bearing_random)

        return SharkState(Z_shark_range, Z_shark_bearing)


    def track_trajectory(self, trajectory):
        """
        Return an Motion_plan_state object representing the trajectory point 0.5 sec ahead
        of current time

        Parameters: 
            trajectory - a list of trajectory points, where each element is 
            a Motion_plan_state object that consist of time stamp, x, y, z,theta
        """
        # determine how ahead should the trajectory point be compared to current time
        look_ahead_time = 0.5

        # only increment the index if it hasn't reached the end of the trajectory list
        while (self.curr_traj_pt_index < len(trajectory)-1) and\
            (self.curr_time + look_ahead_time) > trajectory[self.curr_traj_pt_index].time_stamp: 
                self.curr_traj_pt_index += 1

        return trajectory[self.curr_traj_pt_index]


    def calculate_new_auv_state (self, v, w, delta_t):
        """ 
        Calculate new x, y and theta

        Parameters: 
            v - linear velocity of the robot (m/s)
            w - angular veloctiy of the robot (rad/s)
            delta_t - time step (sec)
        """
        self.x = self.x + v * math.cos(self.theta)*delta_t
        self.y = self.y + v * math.sin(self.theta)*delta_t
        self.theta = angle_wrap(self.theta + w * delta_t)

        self.x_list += [self.x]
        self.y_list += [self.y]
        self.z_list += [self.z]


    def send_trajectory_to_actuators(self, v, w):
        # TODO: For now this should just update AUV States?

        # set time step to 0.1 sec 
        delta_t = 0.1
        self.calculate_new_auv_state(v, w, delta_t)
        

    def log_data(self):
        """
        Print in the terminal (and possibly write the 
        data in a log file?)
        """

        print("AUV [x, y, z, theta]:  [", self.x, ", " , self.y, ", ", self.z, ", ", self.theta, "]")

        # get the latest position from the shark postion lists
        print("Shark [x, y, theta]:  [", self.shark_x_list[-1], ", " , self.shark_y_list[-1], ", ", self.shark_z_list[-1], "]")


    def plot_data(self):
        """
        Plot the position of the robot and the shark
        """
        # plot the new auv position as a red "o"
        self.ax.scatter(self.x, self.y, -10, marker = "o", color='red')
        # draw the lines between the points
        self.ax.plot(self.x_list, self.y_list, self.z_list, color='red')
     
        # plot the new shark position as a blue "o"
        # get the latest position from the shark postion lists
        self.ax.scatter(self.shark_x_list[-1], self.shark_y_list[-1], self.shark_z_list[-1], marker = "x", color="blue")
        # draw the lines between the points
        self.ax.plot(self.shark_x_list, self.shark_y_list, self.shark_z_list, color='red')

        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.5)


    def track_way_point(self, way_point):
        """
        Calculates the v&w to get to the next point along the trajectory

        way_point - a objectState object, represent the trajectory point that we are tracking
        """
        # K_P and v are stand in values
        K_P = 1.0  
        v = 1.0
       
        angle_to_traj_point = math.atan2(way_point.y - self.y, way_point.x - self.x) 
        w = K_P * angle_wrap(angle_to_traj_point - self.theta) #proportional control
        
        return v, w
    

    def get_auv_trajectory(self, v, delta_t):
        """
        Create an array of trajectory points representing a square path

        Parameters:
            v - linear velocity of the robot (m/s)
            delta_t - the time interval between each time stamp (sec)
        """
        traj_list = []
        t = 0
        x = 10
        y = 10
        z = -10

        for i in range(5):
            x = x + v * delta_t
            y = y
            theta = 0
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))

        for i in range(5):
            x = x
            y = y + v * delta_t
            theta = math.pi/2
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))
    
        for i in range(5):
            x = x - v * delta_t
            y = y 
            theta = math.pi
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))

        for i in range(5):
            x = x
            y = y - v * delta_t 
            theta = -(math.pi)/2
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))

        return traj_list


    def load_shark_testing_trajectories(self, filepath):
        """
        Load shark tracking data from the csv file specified by the filepath
        Store all the trajectories in an array of SharkTrajectory objects
            SharkTrajectory contains an array of trajectory points with x and y position of the shark
        
        Parameter:
            filepath - a string, the path to the csv file
        """
        with open(filepath, newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',') 
            line_counter = 0
            x_pos_array = []
            y_pos_array = []

            for row in data_reader:
                # 4 rows are grouped together to represent the states of a shark
                if line_counter % 4 == 0:
                    # row 0 contains the x positions
                    x_pos_array = row
                elif line_counter % 4 == 2:         
                    # row 2 row contains the y positions
                    y_pos_array = row
                    self.shark_testing_trajectories.append(\
                        SharkTrajectory(line_counter//4, x_pos_array, y_pos_array))
                
                # row 1 contains the velocity in x direction
                # row 3 contains the velocity in y direction
                # velocity are not relevant in creating trajectories, so they are ignored
                line_counter += 1


    def main_navigation_loop(self):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data
        """
        
        while True:
            
            auv_sensor_data = self.get_auv_sensor_measurements()
            print("==================")
            print("Curr Auv Sensor Measurements [x, y, z, theta, time]: " +\
                str(auv_sensor_data))
            
            curr_shark = self.get_shark_state()
            curr_shark_x, curr_shark_y, curr_shark_theta = curr_shark.x, curr_shark.y, curr_shark.theta
            print("==================")
            print("Testing get_shark_state [x, y, theta]:  [", curr_shark_x, ", " , curr_shark_y, ", ", curr_shark_theta, "]")

            curr_shark_sensor_measurements = self.get_shark_sensor_measurements(curr_shark_x, curr_shark_y,\
                auv_sensor_data.x, auv_sensor_data.y)
            print("==================")
            print("Curr Shark Sensor Measurements [range, bearing]: " +\
                str(curr_shark_sensor_measurements))

            # test trackTrajectory
            tracking_pt = self.track_trajectory(self.testing_trajectory)
            print("==================")
            print ("Currently tracking point: " + str(tracking_pt))
            
            #v & w to the next point along the trajectory
            print("==================")
            (v, w) = self.track_way_point(tracking_pt)
            print ("v and w: ", v, ", ", w)
            print("====================================")
            print("====================================")

            # update the auv position
            self.send_trajectory_to_actuators(v, w)
            
            # self.log_data()

            self.plot_data()

            # increment the current time by 0.1 second
            self.curr_time += 0.1


def main():
    test_robot = RobotSim(10,10,-10,0.1)
    test_robot.load_shark_testing_trajectories("./data/sharkTrackingData.csv")
    test_robot.main_navigation_loop()


if __name__ == "__main__":
    main()
  

