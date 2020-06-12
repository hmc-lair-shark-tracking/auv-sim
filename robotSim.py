import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import time

# import 3 data representation class
from sharkState import SharkState
from sharkTrajectory import SharkTrajectory
from live3DGraph import Live3DGraph
from motion_plan_state import Motion_plan_state

#import path planning class
from astar import astar
from rrt_dubins import RRT

# keep all the constants in the constants.py file
# to get access to a constant, eg:
#   const.SIM_TIME_INTERVAL
import constants as const


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
    def __init__(self, init_x, init_y, init_z, init_theta, replan_time=0.5, planning_time=1.0):
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
        self.time_array = []
        
        # keep track when there will be new sensor data of sharks
        # start out as 20, so particle filter will get some data in the beginning
        self.sensor_time = const.NUM_ITER_FOR_NEW_SENSOR_DATA

        # index for which trajectory point that we should
        # keep track of
        self.curr_traj_pt_index = 0

        # create a square trajectory (list of motion_plan_state object)
        # with parameter: v = 1.0 m/s and delta_t = 0.5 sec
        self.testing_trajectory = self.get_auv_trajectory(5, 0.5)

        self.live_graph = Live3DGraph()

        #time interval to replan trajectory
        self.replan_time = replan_time
        #time limit for path planning algorithm to find the shortest path
        self.planning_time = planning_time

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
        if self.sensor_time == const.NUM_ITER_FOR_NEW_SENSOR_DATA:
            # iterate through all the sharks that we are tracking
            for shark_id in shark_state_dict: 
                shark_data = shark_state_dict[shark_id]

                delta_x = shark_data.x - auv_sensor_data.x
                delta_y = shark_data.y - auv_sensor_data.y
                
                range_random = np.random.normal(0,5) #Gaussian noise with 0 mean and standard deviation 5
                bearing_random = np.random.normal(0,0.5) #Gaussian noise with 0 mean and standard deviation 0.5

                Z_shark_range = math.sqrt(delta_x**2 + delta_y**2) + range_random
                Z_shark_bearing = angle_wrap(math.atan2(delta_y, delta_x) + bearing_random)

                self.shark_sensor_data_dict[shark_id] = SharkState(shark_data.x, shark_data.y, Z_shark_range, shark_id)
            
            # reset the 2 sec time counter
            self.sensor_time = 0
            
            return True
        else: 
            self.sensor_time += 1

            return False


    def track_trajectory(self, trajectory, new_trajectory):
        """
        Return an Motion_plan_state object representing the trajectory point TRAJ_LOOK_AHEAD_TIME sec ahead
        of current time

        Parameters: 
            trajectory - a list of trajectory points, where each element is 
            a Motion_plan_state object that consist of time stamp, x, y, z,theta
        """
        # only increment the index if it hasn't reached the end of the trajectory list
        if new_trajectory:
            self.curr_traj_pt_index = 0

        while (self.curr_traj_pt_index < len(trajectory)-1) and\
            (self.curr_time + const.TRAJ_LOOK_AHEAD_TIME) > trajectory[self.curr_traj_pt_index].time_stamp: 
            self.curr_traj_pt_index += 1

        return trajectory[self.curr_traj_pt_index]


    def calculate_new_auv_state(self, v, w, delta_t):
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

        self.calculate_new_auv_state(v, w, const.SIM_TIME_INTERVAL)

    def replan_trajectory(self, planner, auv_pos, shark_pos, obstacle, boundary):
        '''after replan_time, calculate a new trajectory based on planner chosen
        
        Parameters:
            planner: path planning algorithm: RRT or A*
            auv_pos: current auv position
            shark_pos: current shark position
            obstacle: obstacle list
            boundary'''
        
        t_end = time.time() + self.planning_time
        shortest_path = []
        shortest_length = float("inf")
        
        if planner == "RRT":
            path_planning = RRT(auv_pos, shark_pos, obstacle, boundary)

        while time.time() < t_end:
            result = path_planning.planning(animation=False)
            if result is not None:
                length = result[0]
                path = result[1]
                if length < shortest_length:
                    shortest_length = length
                    shortest_path = path

        shortest_path.reverse()
        
        step = self.curr_time
        for pt in shortest_path:
            pt.time_stamp = step
            step += const.SIM_TIME_INTERVAL
        
        return shortest_path
    
    def create_trajectory_list(self, traj_list):
        """
        Run this function to update the trajectory list by including intermediate positions 

        Parameters:
            traj_list - a list of Motion_plan_state, represent positions of each node
        """
        time_stamp = 0.1
        
        #constant_gain = 1
    
        trajectory_list = []

        step = 0 

        for i in range(len(traj_list)-1):
            
            trajectory_list.append(traj_list[i])

            x1 = traj_list[i].x
            y1 = traj_list[i].y
            x2 = traj_list[i+1].x 
            y2 = traj_list[i+1].y

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            dist = math.sqrt(dx**2 + dy**2)

            velocity = 1

            time = dist/velocity
            
            counter = 0 

            while counter < time:
                if x1 < x2:
                    x1 += time_stamp * velocity
                if y1 < y2:
                    y1 += time_stamp * velocity
                
                step += time_stamp
                counter += time_stamp
                trajectory_list.append(Motion_plan_state(x1, y1, time_stamp=step))
                
            trajectory_list.append(traj_list[i+1])
            
        return trajectory_list

    def log_data(self):
        """
        Print in the terminal (and possibly write the 
        data in a log file?)
        """

        print("AUV [x, y, z, theta]:  [", self.x, ", " , self.y, ", ", self.z, ", ", self.theta, "]")


    def plot(self, show_live_graph = True, planned_traj_array = [], particle_array = [], obstacle_array = []):
        """
        Wrapper function for plotting and updating shark position

        If we are not showing live graph, it will call the update_shark_location function to 
            update the shark's location without plotting

        Parameters:
            show_live_graph - boolean, determine wheter the live 3d graph should be drawn or not
             planned_traj_array - (optional) an array of trajectories that we want to plot
                each element is an array on its own, where
                    1st element: the planner's name (either "A *" or "RRT")
                    2nd element: the list of Motion_plan_state returned by the planner
            particle_array - (optional) an array of particles
                each element has this format:
                    [x_p, y_p, v_p, theta_p, weight_p]
            obstacle_array - (optional) an array of motion_plan_states that represent the obstacles's
                position and size
        """
        if show_live_graph:
            self.update_live_graph(planned_traj_array, particle_array, obstacle_array)
        else:
            for shark in self.live_graph.shark_array:
                # only update the shark's position without plotting them
                self.live_graph.update_shark_location(shark, self.curr_time)


    def update_live_graph(self, planned_traj_array = [], particle_array = [], obstacle_array = []):
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
            obstacle_array - (optional) an array of motion_plan_states that represent the obstacles's
                position and size
        """
        # scale the arrow for the auv and the sharks properly for graph
        self.live_graph.scale_quiver_arrow()

        self.live_graph.plot_auv(self.x_list, self.y_list, self.z_list)

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

        if obstacle_array != []:
            self.live_graph.plot_obstacles(obstacle_array)

        self.live_graph.ax.legend(self.live_graph.labels)
        
        # re-add the labels because they will get erased
        self.live_graph.ax.set_xlabel('X')
        self.live_graph.ax.set_ylabel('Y')
        self.live_graph.ax.set_zlabel('Z')

        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.5)

        self.live_graph.ax.clear()


    def summary_graphs(self):
        """
        Generate summary plot(s) after the "End Simulation" button is clicked
        """
        # diction where each value stores an array represanting the distance between the auv and a shark
        auv_all_sharks_dist_dict = {}

        for shark in self.live_graph.shark_array:
            # store a list of distances between the auv and the shark at each time-stamp
            dist_array = []
            for i in range(len(self.x_list)-2):
                delta_x = shark.x_pos_array[i] - self.x_list[i]
                delta_y = shark.y_pos_array[i] - self.y_list[i]

                dist_array.append(math.sqrt(delta_x**2 + delta_y**2))
            
            auv_all_sharks_dist_dict[shark.id] = dist_array
        
        # close the 3D simulation plot (if there's any)
        plt.close()

        # plot the distance between auv and sharks over time graph
        self.live_graph.plot_distance(auv_all_sharks_dist_dict, self.time_array)

        plt.show()


    def track_way_point(self, way_point):
        """
        Calculates the v&w to get to the next point along the trajectory

        way_point - a motion_plan_state object, represent the trajectory point that we are tracking
        """
        # K_P and v are stand in values
        K_P = 0.5
        # v = 12
        v = 30  # TODO: currently change it to a very unrealistic value to show the final plot faster
       
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
    

    def check_terminate_cond(self):
        """
        Check if the main navigation loop should terminate automatically
        2 possible terminating condition
            - if the auv and one of the shark is with in the TERMINATE_DISTANCE range
            - if the simulator reaches the MAX_TIME

        Return:
            True - if the main navigation loop should terminate
        """
        reach_any_shark = False
        reach_max_time = False


        if self.curr_time > const.MAX_TIME:
            reach_max_time = True
            return reach_max_time

        for shark in self.live_graph.shark_array:
            delta_x = shark.x_pos_array[-1] - self.x_list[-1]
            delta_y = shark.y_pos_array[-1] - self.y_list[-1]

            distance = math.sqrt(delta_x**2 + delta_y**2)
            
            if distance <= const.TERMINATE_DISTANCE:
                reach_any_shark = True
                break
        
        return reach_any_shark or reach_max_time


    def main_navigation_loop(self, show_live_graph = True):
        """ 
        Wrapper function for the robot simulator
        The loop follows this process:
            getting data -> get trajectory -> send trajectory to actuators
            -> log and plot data

        Parameter:
            show_live_graph - boolean (option), True if the simulator should show the 3D graph
        """
        #set start time
        t_start = self.curr_time

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
            
            # example of how to indicate the obstacles and plot them
            obstacle_array = [Motion_plan_state(757,243, size=2),Motion_plan_state(763,226, size=5)]

            # testing data for plotting RRT_traj
            boundary = [Motion_plan_state(0,0), Motion_plan_state(1000,1000)]

            #condition to replan trajectory
            if self.curr_time == 0 or self.curr_time - t_start >= self.replan_time:
                RRT_traj = self.replan_trajectory("RRT", auv_sensor_data, shark_state_dict[1], obstacle_array, boundary)
                new_trajectory = True
                t_start = self.curr_time
            else:
                new_trajectory = False
            
            # test trackTrajectory
            tracking_pt = self.track_trajectory(RRT_traj, new_trajectory)
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


            # example of first parameter to update_live_graph function
            #planned_traj_array = [["A *", A_star_traj], ["RRT", RRT_traj]]
            planned_traj_array = []

            # testing data for displaying particle array
            particle_array = [[740, 280, 0, 0, 0]]
            
            particle_array += [[740 + i, 280 + np.random.randint(-20, 20, dtype='int'), 0, 0, float(i)/50.0] for i in range(50)]
            
            particle_array += [[740 + np.random.randint(-20, 20, dtype='int'), 280 + np.random.randint(-20, 20, dtype='int'), 0, 0, 0] for i in range(50)]

            # In order to plot your planned trajectory, you have to wrap your trajectory in another array, where
            #   1st element: the planner's name (either "A *" or "RRT")
            #   2nd element: the list of Motion_plan_state returned by your planner
            # Use the "planned_traj_array" as an example
            self.plot(show_live_graph, planned_traj_array, particle_array, obstacle_array)
            
            self.time_array.append(self.curr_time)
            # increment the current time by 0.1 second
            self.curr_time += const.SIM_TIME_INTERVAL

            terminate_loop = self.check_terminate_cond()

            if terminate_loop:
                self.live_graph.run_sim = False
                break

        
        obstacle_array = [Motion_plan_state(757,243, size=10), Motion_plan_state(763,226, size=15)]

        self.live_graph.plot_2d_sim_graph(self.x_list, self.y_list, obstacle_array)

        # "End Simulation" button is pressed, generate summary graphs for this simulation
        # self.summary_plots()


def main():
    test_robot = RobotSim(700,270,0,0.1)
    # load shark trajectories from csv file
    # the second parameter specify the ids of sharks that we want to track

    test_robot.setup("./data/sharkTrackingData.csv", [1])
    test_robot.main_navigation_loop()



if __name__ == "__main__":
    main()
