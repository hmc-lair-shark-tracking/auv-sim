"""a wrapper class to represent states for motion planning
    including x, y, z, theta, v, w, and time stamp"""
class Motion_plan_state:
    #class for motion planning

    def __init__(self,x,y,z=0,theta=0,v=0,w=0, traj_time_stamp=0, plan_time_stamp=0, size=0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.v = v  # linear velocity
        self.w = w  # angular velocity
        self.traj_time_stamp = traj_time_stamp
        self.plan_time_stamp = plan_time_stamp
        self.size = size
        self.parent = None
        self.path = []
        self.length = 0
        self.cost = []

    def __repr__(self):
        #goal location in 2D
        if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return ("[x=" + str(self.x) + ", y="  + str(self.y) + "]")
        #goal location in 3D
        elif self.theta == 0 and self.v == 0 and self.w == 0 and self.size == 0 and self.time_stamp == 0:
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + "]"
        #obstable location in 3D
        elif self.size != 0 and self.traj_time_stamp == 0:
            return "[x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z) + ", size=" + str(self.size) + "]"
        #location for Dubins Path in 2D
        elif self.z ==0 and self.v == 0 and self.w == 0:
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", theta=" + str(self.theta) + ", trag_time=" + str(self.traj_time_stamp) + ", plan_time=" + str(self.plan_time_stamp) + "]"
        else: 
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) +\
                ", theta=" + str(self.theta)  + ", v=" + str(self.v) + ", w=" + str(self.w) +\
                ", trag_time=" + str(self.traj_time_stamp) +  ", plan_time="+  str(self.plan_time_stamp) + "]"


    def __str__(self):
        #goal location in 2D
        if self.z == 0 and self.theta == 0 and self.v == 0 and self.w == 0 and self.time_stamp == 0:
            return ("[x=" + str(self.x) + ", y="  + str(self.y) + "]")
        #goal location in 3D
        elif self.theta == 0 and self.v == 0 and self.w == 0 and self.size == 0 and self.time_stamp == 0:
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) + "]"
        #obstable location in 3D
        elif self.size != 0 and self.traj_time_stamp == 0:
            return "[x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z) + ", size=" + str(self.size) + "]"
        #location for Dubins Path in 2D
        elif self.z ==0 and self.v == 0 and self.w == 0:
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", theta=" + str(self.theta) + ", trag_time=" + str(self.traj_time_stamp) + ", plan_time=" + str(self.plan_time_stamp) + "]"
        else: 
            return "[x=" + str(self.x) + ", y="  + str(self.y) + ", z=" + str(self.z) +\
                ", theta=" + str(self.theta)  + ", v=" + str(self.v) + ", w=" + str(self.w) +\
                ", trag_time=" + str(self.traj_time_stamp) +  ", plan_time="+  str(self.plan_time_stamp) + "]"