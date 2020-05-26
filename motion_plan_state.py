"""a wrapper class to represent states for motion planning
    including x, y, z, theta, v, w, and time stamp"""
class Motion_plan_state:
    #class for motion planning

    def __init__(self,x,y,z = 0,theta = 0,v = 0,w = 0,time_stamp = 0, size = 0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.v = v #linear velocity
        self.w = w #angulr velocity
        self.time_stamp = time_stamp
        self.size = size
        self.parent = None
        self.path = []

    def __repr__(self):
        #goal location in 2D
        if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return ("[" + str(self.x) + ", "  + str(self.y) + "]")
        #goal location in 3D
        elif self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return "[" + str(self.x) + ", "  + str(self.y) + ", " + str(self.z) + "]"
        #obstable location in 3D
        elif self.size != 0:
            return "[" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.size) + "]"
        else: 
            return "[" + str(self.x) + ", "  + str(self.y) + ", " + str(self.z) + ", " +\
                str(self.theta)  + ", " + str(self.v) + ", " + str(self.w) + ", "+  str(self.time_stamp) + "]"

    def __str__(self):
        #goal location in 2D
        if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return "[" + str(self.x) + ", "  + str(self.y) + "]"
        #goal location in 3D
        elif self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return "[" + str(self.x) + ", "  + str(self.y) + ", " + str(self.z) + "]"
        #obstable location in 3D
        elif self.size != 0:
            return "[" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.size) + "]"
        else: 
            return "[" + str(self.x) + ", "  + str(self.y) + ", " + str(self.z) + ", " +\
                str(self.theta)  + ", " + str(self.v) + ", " + str(self.w) + ", "+  str(self.time_stamp) + "]"