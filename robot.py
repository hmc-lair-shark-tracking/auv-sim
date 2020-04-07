import math
import time  # temporarily import time so the code can sleep

class Robot:
    def __init__(self, initX, initY, initTheta):
        self.x = initX
        self.y = initY
        self.theta = initTheta

    def getSharkState(self, v, w, delta):
        self.x = self.x + v * math.cos(self.theta)*delta
        self.y = self.y + v * math.sin(self.theta)*delta
        self.theta = self.theta + w * delta

    def mainNavigationLoop(self):
        while True:
            v = 10
            w = 10
            self.getSharkState(v, w, 0.1)
            print("x postion: ", self.x)
            print("y position: ", self.y)
            print("theta: ", self.theta)
            time.sleep(1)

def main():
    testRobot = Robot(0,0,0.1)
    testRobot.mainNavigationLoop()

if __name__ == "__main__":
    main()