import numpy as np
import math
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geopy.distance 
from motion_plan_state import Motion_plan_state

def create_cartesian(pos, origin): 
    """
    convert a position expressed by latitude and longtitude to a position in cartesian coordinates 
    given a defined origin point 

    Parameter: 
        pos - a tuple of two elements (x,y): x represents latitude; y represents longtitude 
        origin - a tuple of two elements (x,y): x represents latitude; y represents longtitude 
    """

    latitude_origin = origin[0]
    longitude_origin = origin[1]
    latitude = pos[0]
    longtitude = pos[1]

    sign_long = np.sign([longtitude-longitude_origin])
    sign_lat = np.sign([latitude-latitude_origin])
    
    x = sign_long * geopy.distance.vincenty((latitude_origin, longtitude), (latitude_origin, longitude_origin)).m
    y = sign_lat * geopy.distance.vincenty((latitude, longitude_origin), (latitude_origin, longitude_origin)).m

    return (x[0], y[0])

def create_environs(obstacles, boundaries, boats, habitats):
    """
    Given the obstacles, boundaries, boats, and habitats in a system of longtitudes and latitudes
    this function converts them into positions of Cartesian coordinates

    Parameter:
        obstacles: a list of Motion_plan_state objects 
        boundaries: a list of Motion_plan_state objects
        boats: a list of Motion_plan_state objects
        habitats: a list of Motion_plan_state objects
    """

    obstacle_list = []
    boundary_list = []
    boat_list = []
    habitat_list = []

    for obs in obstacles:
        pos = create_cartesian((obs.x, obs.y), ORIGIN_BOUND)
        obstacle_list.append(Motion_plan_state(pos[0], pos[1], size=obs.size))

    for b in boundaries:
        pos = create_cartesian((b.x, b.y), ORIGIN_BOUND)
        boundary_list.append(Motion_plan_state(pos[0], pos[1]))
    
    for boat in boats:
        pos = create_cartesian((boat.x, boat.y), ORIGIN_BOUND)
        boat_list.append(Motion_plan_state(pos[0], pos[1], size=boat.size))

    for habi in habitats:
        pos = create_cartesian((habi.x, habi.y), ORIGIN_BOUND)
        habitat_list.append(Motion_plan_state(pos[0], pos[1], size=habi.size))
    
    return ([obstacle_list, boundary_list, boat_list, habitat_list])

ORIGIN_BOUND = (33.445142, -118.484609)
START = (33.445170, -118.484080)
GOAL = (33.445914, -118.489636)

BOUNDARIES = [Motion_plan_state(33.445914, -118.489636), Motion_plan_state(33.446866, -118.488471),
            Motion_plan_state(33.445064, -118.483723), Motion_plan_state(33.443758, -118.485219),
            Motion_plan_state(33.444783, -118.488223)]

OBSTACLES = [Motion_plan_state(33.445113, -118.484508, size=4.479407446738455),
            Motion_plan_state(33.445101, -118.484462, size=4.337794705821955),
            Motion_plan_state(33.445088, -118.484418, size=4.676061518712341),
            Motion_plan_state(33.445073, -118.484371, size=8.238572668143059),
            Motion_plan_state(33.445047, -118.484288, size=9.775751439799537),
            Motion_plan_state(33.445013, -118.484191, size=8.625835109500015),
            Motion_plan_state(33.444986, -118.484104, size=10.679512620952853),
            Motion_plan_state(33.444951, -118.483997, size=12.150680733968928),
            Motion_plan_state(33.444914, -118.483874, size=13.645304514491206),
            Motion_plan_state(33.444862, -118.483741, size=17.812248298199293),
            Motion_plan_state(33.444779, -118.483577, size=26.601714064649762)]

BOATS = [Motion_plan_state(33.445425, -118.486314, size=5),
        Motion_plan_state(33.444596, -118.485285, size=5),
        Motion_plan_state(33.444412, -118.485508, size=5),
        Motion_plan_state(33.443940, -118.485384, size=5)]  

HABITATS = [Motion_plan_state(33.445733, -118.487789, size=45),
    Motion_plan_state(33.446198, -118.486652, size=32),
    Motion_plan_state(33.445400, -118.485959, size=20),
    Motion_plan_state(33.445287, -118.484928, size=55)]

GOAL_LIST = [(33.444928, -118.484448), (33.444686, -118.484716), 
(33.444328, -118.485606), (33.444811, -118.486454), 
(33.445491, -118.486894), (33.445491, -118.487731), 
(33.446171, -118.488010), (33.446243, -118.488697), (33.445914, -118.489636)]

