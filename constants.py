# the time interval for each iteration of the main navigation loop
# unit: sec
SIM_TIME_INTERVAL = 0.1

# in the track_trajectory function, this indicates
#   how ahead should the trajectory point be compared to current time
# unit: sec
TRAJ_LOOK_AHEAD_TIME = 0.5

# in the get_all_sharks_sensor_measurements function,
#   indicates the number of iteration in the main navigation loop before a new
#   shark sensor data is sent
# eg. in this case: since SIM_TIME_INTERVAL is 0.1 sec, this means that after 2 secs, 
#   there will be new shark sensor data
NUM_ITER_FOR_NEW_SENSOR_DATA = 20


#the time interval moving from one motion_plan_state to another along the path
#unit: sec
PLAN_TIME_INTERVAL = 0.001

# unit: m
TERMINATE_DISTANCE = 1

# unit: sec
MAX_TIME = 2


