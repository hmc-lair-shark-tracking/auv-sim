import math
import time  # temporarily import time so the code can sleep

class RobotSim:
    def __init__(self, initX, initY, initTheta):
        self.x = initX
        self.y = initY
        self.theta = initTheta

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
        data in a log file)
        """
    
    def plotData(self):
        """
        Plot the position of the robot and the shark
        """

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
            self.sendTrajectoryToActuators(v, w)
            
            print("x postion: ", self.x)
            print("y position: ", self.y)
            print("theta: ", self.theta)
            # sleep for 1 seconds before repeating the calculation
            time.sleep(1)

def main():
    testRobot = RobotSim(0,0,0.1)
    testRobot.mainNavigationLoop()

if __name__ == "__main__":
    main()