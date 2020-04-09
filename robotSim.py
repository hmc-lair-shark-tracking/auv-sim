import math
import matplotlib.pyplot as plt
import time  # temporarily import time so the code can sleep


class RobotSim:
    def __init__(self, initX, initY, initTheta):
        # initialize auv's data
        self.x = initX
        self.y = initY
        self.theta = initTheta

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
        sharkTheta = 0
        return (sharkX, sharkY, sharkTheta)

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

        (sharkX, sharkY, sharkTheta)= self.getSharkState()

        print("Shark [x postion, y position, theta]:  [", sharkX, ", " , sharkY, ", ", sharkTheta, "]")
    
    def plotData(self):
        """
        Plot the position of the robot and the shark
        """
        # plot the new auv position as a red "o"
        self.ax.scatter(self.x, self.y, -10, marker = "o", color='red')

        (sharkX, sharkY, sharkTheta)= self.getSharkState()
     
        # plot the new shark position as a blue "o"
        self.ax.scatter(sharkX, sharkY, -15, marker = "x", color="blue")

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
            v = 10
            w = 10

            # update the auv position
            self.sendTrajectoryToActuators(v, w)
            
            self.logData()

            self.plotData()

def main():
    testRobot = RobotSim(10,10,0.1)
    testRobot.mainNavigationLoop()

if __name__ == "__main__":
    main()