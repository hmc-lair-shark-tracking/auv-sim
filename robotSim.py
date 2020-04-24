import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange
import numpy as np 

class RobotSim:
    def __init__(self, initX, initY, initZ, initTheta):
        # initialize auv's data
        self.x = initX
        self.y = initY
        self.theta = initTheta

        # need lists to keep track of all the points if we want to
        #   connect lines between the position points and create trajectories
        self.xList = [initX]
        self.yList = [initY]
        self.zList = [initZ]

        self.sharkXList = []
        self.sharkYList = []
        self.sharkZList = []

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

    def getAuvState(self):
        """
        Return a tuple representing the orientation
        of the robot
        """
        return (self.x, self.y, self.theta)

    def getSharkState(self):
        """
        Return a tuple representing the orientation
        of the shark
        """
        sharkX = 10
        sharkY = 10
        sharkZ = -15
        sharkTheta = 0

        self.sharkXList += [sharkX]
        self.sharkYList += [sharkY]
        self.sharkZList += [sharkZ]

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
        
    def calculateNewAuvState (self, v, w, delta_t):
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

        self.xList += [self.x]
        self.yList += [self.y]
        self.zList += [self.zList[0]]


    def sendTrajectoryToActuators(self, v, w):
        # TODO: For now this should just update AUV States?

        # dummy value: set time step
        delta_t = 0.1
        self.calculateNewAuvState(v, w, delta_t)
        

    def logData(self):
        """
        Print in the terminal (and possibly write the 
        data in a log file?)
        """

        print("AUV [x postion, y position, theta]:  [", self.x, ", " , self.y, ", ", self.theta, "]")

        # get the latest position from the shark postion lists
        print("Shark [x postion, y position, theta]:  [", self.sharkXList[-1], ", " , self.sharkYList[-1], ", ", self.sharkZList[-1], "]")
    
    def plotData(self):
        """
        Plot the position of the robot and the shark
        """
        # plot the new auv position as a red "o"
        self.ax.scatter(self.x, self.y, -10, marker = "o", color='red')
        # draw the lines between the points
        self.ax.plot(self.xList, self.yList, self.zList, color='red')
      
     
        # plot the new shark position as a blue "o"
        # get the latest position from the shark postion lists
        self.ax.scatter(self.sharkXList[-1], self.sharkYList[-1], self.sharkZList[-1], marker = "x", color="blue")
        # draw the lines between the points
        self.ax.plot(self.sharkXList, self.sharkYList, self.sharkZList, color='red')

        plt.draw()

        # pause so the plot can be updated
        plt.pause(1)

    def get_shark_sensor_measurements(self, currSharkX, currSharkY, currAuvX, currAuvY):

        delta_x = currSharkX - currAuvX
        delta_y = currSharkY - currAuvY
        rangeRandom = np.random.normal(0,5)#Gaussian noise with 0 mean and standard deviation 5
        bearingRandom = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

        Z_shark_range = math.sqrt(delta_x**2+delta_y**2) + rangeRandom
        Z_shark_bearing = math.atan2(delta_y, delta_x) + bearingRandom

        return (Z_shark_range, Z_shark_bearing)
    
    def track_way_point(self):
        "calculates the v&w to get to the next point along the trajectory"
        #K_P and v are stand in values
        K_P = 1.0  
        v = 1.0 
        angle_to_traj_point = atan2(way_point.y - self.y, way_point.x - self.x) 
        w = K_P * angle_wrap(angle_to_traj_point - self.yaw) #proportional control
        return v, w
    
    def mainNavigationLoop(self):
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
            
            (currAuvX, currAuvY, currAuvTheta) = self.getAuvState()
            print("==================")
            print("Testing getAuvState [x, y, theta]:  [", currAuvX, ", " , currAuvY, ", ", currAuvTheta, "]")
            
            (currSharkX, currSharkY, currSharkTheta) = self.getSharkState()
            print("Testing getSharkState [x, y, theta]:  [", currSharkX, ", " , currSharkY, ", ", currSharkTheta, "]")
            print("==================")

            # test trackTrajectory
            trackingPt = self.track_trajectory(self.testing_trajectory)
            print ("Currently tracking: ", trackingPt)
            print("==================")
            
            #v&w to the next point along the trajectory
            wayPt = self.track_way_point()
            print ("Currently tracking: ", wayPt)
            print("==================")
            
            (currSharkZRange, currSharkZBearing) = self.get_shark_sensor_measurements(currSharkX, currSharkY, currAuvX, currAuvY)

            # update the auv position
            self.sendTrajectoryToActuators(v, w)
            
            self.logData()

            self.plotData()

            # increment the current time by 0.1 second
            self.curr_time += 0.1

def main():
    testRobot = RobotSim(10,10,-10,0.1)
    testRobot.mainNavigationLoop()

if __name__ == "__main__":
    main()
    
def angle_wrap(ang):
    "takes an angle in radians & sets it between the range of -pi to pi"
    if -math.pi <= ang <= math.pi:
        return ang
    elif ang > math.pi: 
        ang += (-2 * math.pi)
        return angle_wrap(ang)
    elif ang < -math.pi: 
        ang += (2 * math.pi)
        return angle_wrap(ang)
