import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange

class RobotSim:
    def __init__(self, init_x, init_y, init_z, init_theta):
        # initialize auv's data
        self.x = init_x
        self.y = init_y
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

        # assume v = 1 m/s
        # the testing trajectory will travel in a square path that has
        # side length 1 meters. 
        # the robot will start at time_stamp = 0, x = 0, y = 0, theta = 0
        self.testing_trajectory = []

        # the robot only moves in positive x direction
        # (moves to the right)
        for x in arange(0.5, 5.5, 0.5):
            self.testing_trajectory += [[x, x, 0, 0] ]
        # turn the robot, so it heads north
        self.testing_trajectory += [[5.5, 5, 0, math.pi/2.0]]
        # the robot only moves in the positive y direction
        # (moves up)
        for y in arange(0.5, 5.5, 0.5):
            self.testing_trajectory += [[5.5+y, 5, y, math.pi/2.0]]
        # turn the robot, so it heads west
        self.testing_trajectory += [[11, 5, 5, math.pi]]
        # the robot only moves in the negative x direction
        # (moves to the left)
        for x in arange(0.5, 5.5, 0.5):
            self.testing_trajectory += [[11+x, 5-x, 5, math.pi]]
        # turn the robot, so it heads south
        self.testing_trajectory += [[16.5, 0, 5, 3.0*math.pi/2.0]]
        # the robot only moves in the negative y direction
        # (moves down)
        for y in arange(0.5, 5.5, 0.5):
            self.testing_trajectory += [[16.5+y, 5-y, 5, 3.0*math.pi/2.0]]
        # turn the robot, so it heads south
        self.testing_trajectory += [[22, 0, 0, 0]]
        print(self.testing_trajectory)

    def get_auv_state(self):
        """
        Return a tuple representing the orientation
        of the robot
        """
        return (self.x, self.y, self.theta)

    def get_shark_state(self):
        """
        Return a tuple representing the orientation
        of the shark
        """
        sharkX = 10
        sharkY = 10
        sharkZ = -15
        sharkTheta = 0

        self.shark_x_list += [sharkX]
        self.shark_y_list += [sharkY]
        self.shark_z_list += [sharkZ]

        return (sharkX, sharkY, sharkTheta)

    def track_trajectory(self, trajectory):
        """
        Return a list representing the trajectory point 0.5 sec ahead
        of current time

        Parameter: 
            trajectory - a list of trajectory points, where each element is 
            a list that consist of timeStamp x, y, theta
        """
        # determine how ahead should the trajectory point be compared to current time
        look_ahead_time = 0.5
       
        while (self.curr_time + look_ahead_time) > trajectory[self.curr_traj_pt_index][0]:
            # only increment the index if it hasn't reached the end of the trajectory list
            if self.curr_traj_pt_index < len(trajectory):
                self.curr_traj_pt_index += 1

        return trajectory[self.curr_traj_pt_index]
        
    def calculate_new_auv_state (self, v, w, delta_t):
        """ 
        Calculate new x, y and theta

        Parameters: 
            v - linear velocity of the robot
            w - angular veloctiy of the robot
            delta_t - time step
        """
        self.x = self.x + v * math.cos(self.theta)*delta_t
        self.y = self.y + v * math.sin(self.theta)*delta_t
        self.theta = self.theta + w * delta_t

        self.x_list += [self.x]
        self.y_list += [self.y]
        self.z_list += [self.z_list[0]]


    def send_trajectory_to_actuators(self, v, w):
        # TODO: For now this should just update AUV States?

        # dummy value: set time step
        delta_t = 0.1
        self.calculate_new_auv_state(v, w, delta_t)
        

    def log_data(self):
        """
        Print in the terminal (and possibly write the 
        data in a log file?)
        """

        print("AUV [x postion, y position, theta]:  [", self.x, ", " , self.y, ", ", self.theta, "]")

        # get the latest position from the shark postion lists
        print("Shark [x postion, y position, theta]:  [", self.shark_x_list[-1], ", " , self.shark_y_list[-1], ", ", self.shark_z_list[-1], "]")
    
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
        plt.pause(1)


    def main_navigation_loop(self):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data
        """
        
        while True:
            # dummy values for linear and angular velocity
            v = 1  # m/s
            w = 0.5   # rad/s
            
            (currAuvX, currAuvY, currAuvTheta) = self.get_auv_state()
            print("==================")
            print("Testing get_auv_state [x, y, theta]:  [", currAuvX, ", " , currAuvY, ", ", currAuvTheta, "]")
            
            (currSharkX, currSharkY, currSharkTheta) = self.get_shark_state()
            print("Testing get_shark_state [x, y, theta]:  [", currSharkX, ", " , currSharkY, ", ", currSharkTheta, "]")
            print("==================")

            # test trackTrajectory
            trackingPt = self.track_trajectory(self.testing_trajectory)
            print ("Currently tracking: ", trackingPt)
            print("==================")

            # update the auv position
            self.send_trajectory_to_actuators(v, w)
            
            self.log_data()

            self.plot_data()

            # increment the current time by 0.1 second
            self.curr_time += 0.1

def main():
    testRobot = RobotSim(10,10,-10,0.1)
    testRobot.main_navigation_loop()

if __name__ == "__main__":
    main()