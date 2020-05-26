import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

# import 3 data representation class
from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
from live3DGraph import Live3DGraph
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

        self.shark_sensor_data_dict = {}

        # keep track of the current time that we are in
        # each iteration in the while loop will be assumed as 0.1 sec
        self.curr_time = 0
        
        # keep track when there will be new sensor data of sharks
        # start out as 20, so particle filter will get some data in the beginning
        self.sensor_time = 20

        # index for which trajectory point that we should
        # keep track of
        self.curr_traj_pt_index = 0

        # create a square trajectory (list of motion_plan_state object)
        # with parameter: v = 1.0 m/s and delta_t = 0.5 sec
        self.testing_trajectory = self.get_auv_trajectory(5, 0.5)

        self.live_graph = Live3DGraph()


    def get_auv_state(self):
        """
        Return a Motion_plan_state representing the orientation and the time stamp
        of the robot
        """
        return Motion_plan_state(self.x, self.y, theta = self.theta, time_stamp=self.curr_time)


    def get_all_sharks_state(self):
        """
        Return a dictionary representing state for all the sharks 
            key = id of the shark & value = the shark's position (stored as a Motion_plan_state)
        """

        # using dictionary so we can access the state of a shark based on its id quickly?
        shark_state_dict = {}

        for shark in self.live_graph.shark_array:
            shark_state_dict[shark.id] = shark.get_curr_position()

        return shark_state_dict


    def get_auv_sensor_measurements(self):
        """
        Return an Motion_plan_state object that represents the measurements
            of the auv's x,y,z,theta position with a time stamp
        The measurement has random gaussian noise
        """
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution

        # np.random.normal returns a single sample drawn from the parameterized normal distribution
        # we actually omitted the third parameter which determines the number of samples that we would like to draw

        return Motion_plan_state(x = self.x + np.random.normal(0,1),\
            y = self.y + np.random.normal(0,1),\
            z = self.z + np.random.normal(0,1),\
            theta = angle_wrap(self.theta + np.random.normal(0,1)),\
            time_stamp = self.curr_time)


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
        if self.sensor_time == 20:
            # iterate through all the sharks that we are tracking
            for shark_id in shark_state_dict: 
                shark_data = shark_state_dict[shark_id]

                delta_x = shark_data.x - auv_sensor_data.x
                delta_y = shark_data.y - auv_sensor_data.y
                
                range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
                bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

                Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
                Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) + bearing_random)

                self.shark_sensor_data_dict[shark_id] = SharkState(Z_shark_range, Z_shark_bearing, shark_id)
            
            # reset the 2 sec time counter
            self.sensor_time = 0
            
            return True
        else: 
            self.sensor_time += 1
            return False


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


    def update_live_graph(self, planned_traj_array = [], particle_array = []):
        """
        Plot the position of the robot, the sharks, and any planned trajectories

        Parameter: 
            planned_traj_array - (optional) an array of trajectories that we want to plot
                each element is an array on its own, where
                    1st element: the planner's name (either "A *" or "RRT")
                    2nd element: the list of Motion_plan_state returned by the planner
            particle_array - (optional) an array of particles
                each element has this format:
                    [x_p, y_p, v_p, theta_p, weight_p]
        """
        
        # plot the new auv position as a red "o"
        self.live_graph.ax.plot(self.x_list, self.y_list, self.z_list,\
            marker = 'o', linestyle = '-', color = 'red', label='auv')

        # plot the new positions for all the sharks that the robot is tracking
        self.live_graph.plot_sharks(self.curr_time)
        
        # if there's any planned trajectory to plot, plot each one
        if planned_traj_array != []:
            for planned_traj in planned_traj_array:
                # pass in the planner name and the trajectory array
                self.live_graph.plot_planned_traj(planned_traj[0], planned_traj[1])

        # if there's particles to plot, plot them
        if particle_array != []:
            self.live_graph.plot_particles(particle_array)

        self.live_graph.ax.legend(self.live_graph.labels)
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.5)

        self.live_graph.ax.clear()


    def track_way_point(self, way_point):
        """
        Calculates the v&w to get to the next point along the trajectory

        way_point - a motion_plan_state object, represent the trajectory point that we are tracking
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
        x = 760
        y = 300
        z = -10

        for i in range(20):
            x = x + v * delta_t
            y = y
            theta = 0
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))

        for i in range(20):
            x = x
            y = y + v * delta_t
            theta = math.pi/2
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))
    
        for i in range(20):
            x = x - v * delta_t
            y = y 
            theta = math.pi
            t = t + delta_t

            traj_list.append(Motion_plan_state(x,y,z,theta,time_stamp=t))

        for i in range(20):
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
        shark_testing_trajectories = []

        with open(filepath, newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',') 
            line_counter = 0
            x_pos_array = []
            x_vel_array = []
            y_pos_array = []
            y_vel_array = []

            for row in data_reader:
                # 4 rows are grouped together to represent the states of a shark
                if line_counter % 4 == 0:
                    # row 0 contains the x position
                    x_pos_array = row
                elif line_counter % 4 == 1:
                     # row 1 contains the x velocity
                    x_vel_array = row
                elif line_counter % 4 == 2:         
                    # row 2 row contains the y positions
                    y_pos_array = row
                elif line_counter % 4 == 3:
                    y_vel_array = row
                    shark_testing_trajectories.append(\
                        SharkTrajectory(line_counter//4, x_pos_array, y_pos_array, x_vel_array, y_vel_array))
                
                # row 1 contains the velocity in x direction
                # row 3 contains the velocity in y direction
                # velocity are not relevant in creating trajectories, so they are ignored
                line_counter += 1
        
        return shark_testing_trajectories


    def setup(self, data_filepath, shark_id_array = []):
        """
        Run this function if we want to track sharks based on their trajectory data in csv file

        Parameters:
            data_filepath - a string, represent the path the csv data file
            shark_id_array - an array indicating the id of sharks we want to track
                eg. for the sharkTrackingData.csv (with 32 sharks), the available ids have the range [0, 31]
        """
        # load the array of 32 shark trajectories for testing
        shark_testing_trajectories = self.load_shark_testing_trajectories(data_filepath)
        
        # based on the id of the shark, build an array of shark that we will track 
        # for this simulation
        self.live_graph.shark_array = list(map(lambda i: shark_testing_trajectories[i],\
            shark_id_array))
        
        self.live_graph.load_shark_labels()
       

    def main_navigation_loop(self):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data
        """
        
        while self.live_graph.run_sim:
            
            auv_sensor_data = self.get_auv_sensor_measurements()
            print("==================")
            print("Curr Auv Sensor Measurements [x, y, z, theta, time]: " +\
                str(auv_sensor_data))
  
            shark_state_dict = self.get_all_sharks_state()
            print("==================")
            print("All the Shark States [x, y, ..., time_stamp]: " + str(shark_state_dict))

            has_new_data = self.get_all_sharks_sensor_measurements(shark_state_dict, auv_sensor_data)

            if has_new_data == True:
                print("======NEW DATA=======")
                print("All The Shark Sensor Measurements [range, bearing]: " +\
                    str(self.shark_sensor_data_dict))

            # test trackTrajectory
            tracking_pt = self.track_trajectory(self.testing_trajectory)
            print("==================")
            print ("Currently tracking point: " + str(tracking_pt))
            
            #v & w to the next point along the trajectory
            (v, w) = self.track_way_point(tracking_pt)
            print("==================")
            print ("v and w: ", v, ", ", w)
            print("====================================")
            print("====================================")

            # update the auv position
            self.send_trajectory_to_actuators(v, w)
            
            # self.log_data()

            # testing data for plotting A_star_traj
            A_star_traj = [Motion_plan_state(740, 280)]
            A_star_traj += [Motion_plan_state(740+i, 280+i) for i in range(50)]

            # testing data for plotting RRT_traj
            RRT_traj = [Motion_plan_state(760, 230)]
            RRT_traj += [Motion_plan_state(760+i, 230+i) for i in range(50)]
            
            # example of first parameter to update_live_graph function
            planned_traj_array = [["A *", A_star_traj], ["RRT", RRT_traj]]

            # testing data for displaying particle array
            particle_array = [[740, 280, 0, 0, 0]]
            
            particle_array += [[740 + np.random.randint(-20, 20, dtype='int'), 280 + np.random.randint(-20, 20, dtype='int'), 0, 0, 0] for i in range(50)]
            
            # In order to plot your planned trajectory, you have to wrap your trajectory in another array, where
            #   1st element: the planner's name (either "A *" or "RRT")
            #   2nd element: the list of Motion_plan_state returned by your planner
            # Use the "planned_traj_array" as an example
            self.update_live_graph(planned_traj_array, particle_array)
            
            # increment the current time by 0.1 second
            self.curr_time += 0.1

        print("end this while loop now!")


def main():
    test_robot = RobotSim(740,280,-10,0.1)
    # load shark trajectories from csv file
    # the second parameter specify the ids of sharks that we want to track
    test_robot.setup("./data/sharkTrackingData.csv", [1, 2])
    test_robot.main_navigation_loop()


if __name__ == "__main__":
    main()
  

