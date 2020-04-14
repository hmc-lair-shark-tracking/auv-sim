import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


    def trackTrajectory(self, trajectory):
        """
        Return a list representing the trajectory point 0.5 sec ahead
        of current time

        Parameter: 
            trajectory - a list of trajectory points, where each element is 
            a list that consist of timeStamp x, y, theta
        """
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


    def mainNavigationLoop(self):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data
        """
        
        while True:
            # dummy values for linear and angular velocity
            v = 2  # m/s
            w = 0.5   # rad/s
            
            (currAuvX, currAuvY, currAuvTheta) = self.getAuvState()
            print("==================")
            print("Testing getAuvState [x, y, theta]:  [", currAuvX, ", " , currAuvY, ", ", currAuvTheta, "]")
            
            (currSharkX, currSharkY, currSharkTheta) = self.getSharkState()
            print("Testing getSharkState [x, y, theta]:  [", currSharkX, ", " , currSharkY, ", ", currSharkTheta, "]")
            print("==================")

            # update the auv position
            self.sendTrajectoryToActuators(v, w)
            
            self.logData()

            self.plotData()

def main():
    testRobot = RobotSim(10,10,-10,0.1)
    testRobot.mainNavigationLoop()

if __name__ == "__main__":
    main()