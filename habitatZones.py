# add all the stuff we gotta import
import math
import decimal
import time
import numpy as np
import random
from copy import copy, deepcopy
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import uniform 
from numpy.random import randn
import threading
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point, LinearRing, LineString, MultiPolygon
#from particle import Particle
from live3DGraph import Live3DGraph
#from twoDfigure import Figure
from motion_plan_state import Motion_plan_state
from robotSim import RobotSim
from sharkTrajectory import SharkTrajectory

def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians

    USE ANY TIME THERE IS AN ANGLE CALCULATION
    """
    if -math.pi <= ang <= math.pi:
        return ang
    elif ang > math.pi: 
        ang += (-2 * math.pi)
        return angle_wrap(ang)
    elif ang < -math.pi: 
        ang += (2 * math.pi)
        return angle_wrap(ang)

def velocity_wrap(velocity):
    if velocity <= 5:
        return velocity  
    elif velocity > 5: 
        velocity += -5
        return velocity_wrap(velocity)

class Particle: 
        def __init__(self, x_shark, y_shark):
            #set L (side length of square that the random particles are in) and N (number of particles)
            INITIAL_PARTICLE_RANGE = 150
            NUMBER_OF_PARTICLES = 1000
            #particle has 5 properties: x, y, velocity, theta, weight (starts at 1/N)
            self.x_p = x_shark + random.uniform(-INITIAL_PARTICLE_RANGE, INITIAL_PARTICLE_RANGE)
            self.y_p = y_shark + random.uniform(-INITIAL_PARTICLE_RANGE, INITIAL_PARTICLE_RANGE)
            self.v_p = random.uniform(0, 5)
            self.theta_p = random.uniform(-math.pi, math.pi)
            self.weight_p = 1/NUMBER_OF_PARTICLES

        def update_particle(self, dt):
            """
                updates the particles location with random v and theta

                input (dt) is the amount of time the particles are "moving" 
                    generally set to .1, but it should be whatever the "time.sleep" is set to in the main loop

            """

            #random_v and random_theta are values to be added to the velocity and theta for randomization
            RANDOM_VELOCITY = 5
            RANDOM_THETA = math.pi/2
            #change velocity & pass through velocity_wrap
            self.v_p += random.uniform(0, RANDOM_VELOCITY)
            self.v_p = velocity_wrap(self.v_p)
            #change theta & pass through angle_wrap
            self.theta_p += random.uniform(-RANDOM_THETA, RANDOM_THETA)
            self.theta_p = angle_wrap(self.theta_p)
            #change x & y coordinates to match 
            self.x_p += self.v_p * math.cos(self.theta_p) * dt
            self.y_p += self.v_p * math.sin(self.theta_p) * dt
            
            
        def calc_particle_alpha(self, x_auv, y_auv, theta_auv):
            """
                calculates the alpha value of a particle
            """
            particleAlpha = angle_wrap(math.atan2((-y_auv + self.y_p), (self.x_p + -x_auv)) - theta_auv)
            return particleAlpha

        def calc_particle_range(self, x_auv, y_auv):
            """
                calculates the range from the particle to the auv
            """
            particleRange_squared = (y_auv - self.y_p)**2 + (x_auv - self.x_p)**2
            return particleRange_squared

        def weight(self, auv_alpha, particleAlpha, auv_range, particleRange):
            """
                calculates the weight according to alpha, then the weight according to range
                they are multiplied together to get the final weight
            """
            #alpha weight
            SIGMA_ALPHA = .5
            constant = 2.506628275
            MINIMUM_WEIGHT = .001
            if particleAlpha > 0:
                function_alpha = .001 + (1/(SIGMA_ALPHA * constant)* (math.e**(((-((angle_wrap(float(particleAlpha) - auv_alpha[0])**2))))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = function_alpha
            elif particleAlpha == 0:
                function_alpha = .001 + (1/(SIGMA_ALPHA * constant)* (math.e**(((-((angle_wrap(float(particleAlpha) - auv_alpha[0])**2))))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = function_alpha
            else:
                function_alpha = .001 + (1/(SIGMA_ALPHA * constant)* (math.e**(((-((angle_wrap(float(particleAlpha) - auv_alpha[0])**2))))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = function_alpha
    
            #range weight
            SIGMA_RANGE = 100
            function_weight =  MINIMUM_WEIGHT + (1/(SIGMA_RANGE * constant)* (math.e**(((-((particleRange - auv_range)**2)))/(2*(SIGMA_RANGE)**2))))
            
            #multiply weights
            self.weight_p = function_weight * self.weight_p

        

class Shark: 
    def __init__(self, x_shark, y_shark):
        #set L (side length of square that the random particles are in) and N (number of particles)
        INITIAL_SHARK_RANGE = 150
        self.x_shark = x_shark
        self.y_shark = y_shark
        self.v_shark = random.uniform(0, 5)
        self.theta_shark = random.uniform(-math.pi, math.pi)
        self.xy_coordinates = [[self.x_shark, self.y_shark]]

    def update_shark(self, dt): 
        #updates shark position and randomly changes velocity and theta
        
        self.x_shark = self.x_shark + dt * (self.v_shark * math.cos(self.theta_shark))
        self.y_shark = self.y_shark + dt * (self.v_shark * math.sin(self.theta_shark))
        self.v_shark += random.uniform(-2, 2)
        self.v_shark = velocity_wrap(self.v_shark)
        self.theta_shark += random.uniform(-math.pi/4, math.pi/4)
        self.theta_shark = angle_wrap(self.theta_shark)
        self.xy_coordinates.append([self.x_shark, self.y_shark])
        return [self.x_shark, self.y_shark]

    
    
    def find_closest_hotspot(self, hotspots):
        list_of_distances_squared = []
        for hotspot in hotspots:
            distance_squared = (hotspot[1] - self.y_shark)**2 + (hotspot[0] - self.x_shark)**2
            list_of_distances_squared.append(distance_squared)
        closest_distance = list_of_distances_squared.index(min(list_of_distances_squared))
        return closest_distance


    def update_shark_hotspots(self, hotspots, percent, dt):
        index = self.find_closest_hotspot(hotspots)
        hotspot = hotspots[index]
        theta_to_hotspot = angle_wrap(math.atan2((-hotspot[1] + self.y_shark), (self.x_shark + -hotspot[0])) - self.theta_shark)
        chance = random.uniform(0,100)
        if chance < percent:
            self.theta_shark = theta_to_hotspot
        else:
            self.theta_shark += random.uniform(-math.pi/4, math.pi/4)
            self.theta_shark = angle_wrap(self.theta_shark)
        self.x_shark = self.x_shark + dt * (self.v_shark * math.cos(self.theta_shark))
        self.y_shark = self.y_shark + dt * (self.v_shark * math.sin(self.theta_shark))
        self.v_shark += random.uniform(-2, 2)
        self.v_shark = velocity_wrap(self.v_shark)
        self.xy_coordinates.append([self.x_shark, self.y_shark])
        return [self.x_shark, self.y_shark]

    

def main_shark_traj_function():
    NUMBER_OF_SHARKS = 3
    GRID_RANGE = 150
    #half the distance of the square where the shark's initial location will be randomized
    NUMBER_OF_TIMESTAMPS = 300
    #number of loops/timestamps the function will run
    dt = 1
    #length of each timestamp, in seconds
    percent = 50
    #percent of the time the sharks will swim towards the closest hotspot to them
    sharks = []
    coordinates = []
    hotspots_present = True
    #true or false for presence of hotspots
    for x in range(1,NUMBER_OF_SHARKS):
        shark = Shark(random.uniform(-GRID_RANGE, GRID_RANGE), random.uniform(-GRID_RANGE, GRID_RANGE))
        sharks.append(shark)

    if hotspots_present == False:
        for x in range(1, NUMBER_OF_TIMESTAMPS):
            for shark in sharks: 
                coordinate = shark.update_shark(dt)
                coordinates.append(coordinate)
    
    if hotspots_present == True: 
        hotspots = []
        hotspot1 = [0,0]
        hotspots.append(hotspot1)
        hotspot2 = [50,0]
        hotspots.append(hotspot2)
        for x in range(1, NUMBER_OF_TIMESTAMPS):  
            for shark in sharks: 
                coordinate = shark.update_shark_hotspots(hotspots, percent, dt)
                coordinates.append(coordinate)

    key = 1
    shark_dict = {}
    for i in range(len(coordinates)):
        shark_dict[i] = Motion_plan_state(x = float(coordinates[i][0]), y = float(coordinates[i][1]), traj_time_stamp = i * 0.03)
    print(shark_dict)




class ParticleFilter:
    # 2 sets of initial data- shark's initial position and velocity, and position of AUV 
    # output- estimates the sharks position and velocity
    def __init__(self, init_theta, init_x_auv_1, init_y_auv_1, init_x_auv_2, init_y_auv_2, init_theta_2):
        "how you create an object out of the particle Filter class i.e. "
        self.theta = init_theta
        self.x_auv = init_x_auv_1
        self.y_auv = init_y_auv_1
        self.x_auv_2 = init_x_auv_2
        self.y_auv_2 = init_y_auv_2
        self.theta_2 = init_theta_2

    def update_auv(self, dt):
        #v_x_shark = random.uniform(-5, 5)
        #v_y_shark = random.uniform(-5, 5)
        v_x_auv = 1
        v_y_auv = 0
        self.x_auv = self.x_auv + (v_x_auv * dt)
        self.y_auv = self.y_auv + (v_y_auv * dt)
        return [self.x_auv, self.y_auv]
    
    def update_auv_2(self, dt):
        #v_x_shark = random.uniform(-5, 5)
        #v_y_shark = random.uniform(-5, 5)
        v_x_auv = 1
        v_y_auv = 0
        self.x_auv_2 = self.x_auv_2 + (v_x_auv * dt)
        self.y_auv_2 = self.y_auv_2 + (v_y_auv * dt)
        return [self.x_auv_2, self.y_auv_2]

    def updateShark(self, dt):
        #v_x_shark = random.uniform(-5, 5)
        #v_y_shark = random.uniform(-5, 5)
        v_x_shark = 1
        v_y_shark = 1
        self.x_shark = self.x_shark + (v_x_shark * dt)
        self.y_shark = self.y_shark + (v_y_shark * dt)
        return [self.x_shark, self.y_shark]

    def auv_to_alpha(self, x_shark, y_shark):
        #calculates auv's alpha from the shark
        list_of_real_alpha = []
        real_alpha = angle_wrap(math.atan2((-self.y_auv + y_shark), (x_shark- self.x_auv)) - self.theta)
        #real_alpha_2 = angle_wrap(math.atan2((-self.y_auv_2 + self.y_shark), (self.x_shark- self.x_auv_2)) - self.theta_2)
        list_of_real_alpha.append(real_alpha)
        list_of_real_alpha.append(-real_alpha)
        return list_of_real_alpha
    
    def auv_to_alpha_2(self, x_shark, y_shark):
        #calculates auv's alpha from the shark
        list_of_real_alpha = []
        real_alpha_2 = angle_wrap(math.atan2((-self.y_auv_2 + y_shark), (x_shark- self.x_auv_2)) - self.theta_2)
        list_of_real_alpha.append(real_alpha_2)
        list_of_real_alpha.append(-real_alpha_2)
        return list_of_real_alpha

    def range_auv(self, x_shark, y_shark):
        range = []
        range_value = (float(self.y_auv)- y_shark)**2 + (float(self.x_auv)-float(x_shark))**2
        #print("range of auv is")
        #print(range)
        return range_value
    
    def range_auv_2(self, x_shark, y_shark):
        range = []
        range_value = (float(self.y_auv_2)- y_shark)**2 + (float(self.x_auv_2)-float(x_shark))**2
        range.append(range_value)
        #print("range of auv is")
        #print(range)
        return range_value
        
    def normalize(self, weights_list, weights_list_2):
        newlist = []
        newlist_2 = []
        newlist_3 = []
        denominator= max(weights_list)
        denominator_2 = max(weights_list_2)
        for weight in weights_list:
            #weight1 = (1/ denominator) * weight
            newlist.append(weight)
        for weight in weights_list_2:
            #weight2 = (1/ denominator_2) * weight
            newlist_2.append(weight)
        index = -1
        final_list_of_weights = []
        for weight in newlist:
            index += 1
            new_weight = weight + newlist[index]
            final_list_of_weights.append(new_weight)
        final_denominator = max(final_list_of_weights)
        normalized_list = []
        for weight in final_list_of_weights:
            weight_final = (1/ final_denominator) * weight
            normalized_list.append(weight_final)
        return normalized_list

    def particleMean(self, new_particles):
        """caculates the mean of the particles x and y positions"""
        sum_x = 0
        sum_y = 0
        x_mean = 0
        y_mean = 0
        count = 0
        for particle in new_particles:
            sum_x += particle.x_p
            sum_y += particle.y_p
            count += 1
        x_mean = sum_x/count
        y_mean = sum_y/ count
        xy_mean = [x_mean, y_mean]
        return xy_mean

    def meanError(self, x_mean, y_mean):
        
        x_difference = x_mean - self.x_shark
        y_difference = y_mean - self.y_shark
        range_error = math.sqrt((x_difference**2) + (y_difference **2))
        #alpha_error = math.atan2(y_difference, x_difference)
        #print("error")
        #print(range_error)
        return (range_error)

    def correct(self, normalize_list, old_coordinates):
        list_of_new_particles = []
        new_particles = []
        copy = []
        count = -1
        #print(normalize_list)
        for particle in old_coordinates:
            if particle.weight_p < 0.2:
                count += 1
                #print(particle.weight_p)
                copy = deepcopy(particle)
                list_of_new_particles.append(copy)
                    
                    #list_of_new_particles.append(copy)
                    #print("count, ", count, "x, y", old_coordinates[count][:2] )
                    #print(list_of_new_particles)
            elif particle.weight_p < 0.4:
                #print(particle.weight_p)
                count += 1
                copy1 = deepcopy(particle)
                list_of_new_particles.append(copy1)
                copy2 = deepcopy(particle)
                list_of_new_particles.append(copy2)
                    
                    #print(list_of_new_particles)
            elif particle.weight_p < 0.6:
                #print(particle.weight_p)
                count += 1
                copy3 = deepcopy(particle)
                list_of_new_particles.append(copy3)
                copy4 = deepcopy(particle)
                list_of_new_particles.append(copy4)                    
                copy5 = deepcopy(particle)
                list_of_new_particles.append(copy5)
                    

                    #print(list_of_new_particles)
            elif particle.weight_p < .8:
                #print(particle.weight_p)
                count += 1
                copy6 = deepcopy(particle)
                list_of_new_particles.append(copy6)
                copy7 = deepcopy(particle)
                list_of_new_particles.append(copy7)
                copy8 = deepcopy(particle)
                list_of_new_particles.append(copy8)
                copy9 = deepcopy(particle)
                list_of_new_particles.append(copy9)
                    
                    
                    #print(list_of_new_particles)
            elif particle.weight_p <= 1.0:
                #print(particle.weight_p)
                count += 1
                copy10 = deepcopy(particle)
                list_of_new_particles.append(copy10)
                copy11 = deepcopy(particle)
                list_of_new_particles.append(copy11)
                copy12 = deepcopy(particle)
                list_of_new_particles.append(copy12)
                copy13 = deepcopy(particle)
                list_of_new_particles.append(copy13)
                copy14 = deepcopy(particle)
                list_of_new_particles.append(copy14)
            else:
                print("something is not right with the weights")
        #print('list of new particles')
        #print(list_of_new_particles)
        
        #for particle in list_of_new_particles: 
           # print("x:", particle.x_p, " y:", particle.y_p, " velocity:", particle.v_p, " theta:", particle.theta_p, " weight:", particle.weight_p)
        #print(count)
        for n in range(len(normalize_list)): 
            x = random.choice(len(list_of_new_particles))
            new_particles.append(list_of_new_particles[x])
        #print("new particles")
        #print(new_particles)
        return new_particles
    
    def particle_coordinates(self, particles): 
        particle_coordinates = []
        individual_coordinates = []
        for particle in particles:
            individual_coordinates.append(particle.x_p)
            individual_coordinates.append(particle.y_p)
            individual_coordinates.append(particle.weight_p)
            #print("individual coordinates", individual_coordinates)
            particle_coordinates.append(individual_coordinates)
            individual_coordinates = []
            #print("particle_coordinates", particle_coordinates)
        return particle_coordinates
    
    def cluster_over_time_function(self, particles, actual_shark_coordinate_x, actual_shark_coordinate_y, sim_time, list_of_error_mean):
        list_of_answers = []
        count = 0
        for particle in particles:
            sum = math.sqrt(((particle.x_p - actual_shark_coordinate_x[-1])**2)  + ((particle.y_p - actual_shark_coordinate_y[-1])**2))
            if sum <= 1.1* (list_of_error_mean[9]):
                count += 1
            if count == 560:
                initial_time = sim_time
                list_of_answers.append(sim_time)
            """
            elif count == 0:
                difference = sim_time - initial_time
                if difference >= 1:
                    list_of_answers.append(sim_time)
            """
            return list_of_answers
    
    def create_and_update(self, particles):
        for particle in particles: 
            particle.update_particle(.1)
            #print("x:", particle.x_p, " y:", particle.y_p, " velocity:", particle.v_p, " theta:", particle.theta_p, " weight:", particle.weight_p)

    def update_weights(self, particles, auv_alpha, auv_range, auv_alpha_2, auv_range_2):
        #print("auv range and alpha", auv_alpha, auv_range)

        
        for particle in particles: 
            particleAlpha = particle.calc_particle_alpha(self.x_auv, self.y_auv, self.theta)
            particleRange = particle.calc_particle_range(self.x_auv, self.y_auv)
            particle.weight(auv_alpha, particleAlpha, auv_range, particleRange)
            #print("weight: ", particle.weight_p)
        list_of_weights = []
        for particle in particles: 
            list_of_weights.append(particle.weight_p)
        #print("new y in the loop", self.y_auv_2)
        for particle in particles: 
            particleAlpha_2 = particle.calc_particle_alpha(self.x_auv_2, self.y_auv_2, self.theta_2)
            particleRange_2 = particle.calc_particle_range(self.x_auv_2, self.y_auv_2)
            particle.weight(auv_alpha_2, particleAlpha_2, auv_range_2, particleRange_2)
            #print("weight: ", particle.weight_p)
        list_of_weights_2 = []
        for particle in particles: 
            list_of_weights_2.append(particle.weight_p)

        normalized_weights = self.normalize(list_of_weights, list_of_weights_2)
        return normalized_weights
        
    def final_weight_correct(self, particles, normalized_weights1, normalized_weights2):
        final_normalized_weights = []
        for x in range(len(normalized_weights1)):
            if normalized_weights1[x] >= normalized_weights2[x]: 
                    final_normalized_weights.append(normalized_weights1[x])
            if normalized_weights1[x] < normalized_weights2[x]: 
                    final_normalized_weights.append(normalized_weights2[x])

        count = 0
        for particle in particles: 
            particle.weight_p = final_normalized_weights[count]
            #print("new normalized particle weight: ", particle.weight_p, "aka i work")
            count += 1
        new_particles = self.correct(final_normalized_weights, particles)
        
        particles = new_particles
        #for particle in particles: 
            #print("x:", particle.x_p, " y:", particle.y_p, " velocity:", particle.v_p, " theta:", particle.theta_p, " weight:", particle.weight_p)
        
        return particles

    

class HabitatZones: 
    def __init__(self, grid_range, cell_size):
        # grid range is 150 m
        # boundary is the boundary of area that must be split into cells, has a min x/y and max x/y
        # cell size is 10m (length of each size)
        self.grid_range = grid_range
        self.cell_size = cell_size
        
        NUMBER_OF_ZONES = (self.grid_range/10)**2
    
    def create_zones(self, zone_width):
        NUMBER_OF_PARTICLES = 1000
        label = 0
        #label should go from top left to top right and then down a row to repeat
        minx = -self.grid_range
        maxx = -self.grid_range + self.cell_size
        miny = self.grid_range - self.cell_size
        maxy = self.grid_range
        probability = 1/NUMBER_OF_PARTICLES
        zones = []
        
        zones = [[0 for x in range(int(math.ceil(self.grid_range * 2) / self.cell_size))] for y in range(int(math.ceil(self.grid_range * 2) / self.cell_size))]

        for i in range(len(zones)):
            for j in range(len(zones[0])):
                zones[i][j] = [(minx, maxx), (miny, maxy), label, probability]
                minx += 10
                maxx += 10
                if maxx >= int(self.grid_range +1):
                    maxx = -self.grid_range + 10
                    minx = -self.grid_range
                #print(zones[i][j])
                label += 1
               
            maxy += -10
            miny += -10
        return zones

    def indexToCell(self, x_p, y_p, zones):
        #round x_p and y_p up to the nearest 10
        if x_p > self.grid_range:
            x_p = self.grid_range
        if x_p < -self.grid_range:
            x_p = -self.grid_range
        if y_p > self.grid_range:
            y_p = self.grid_range
        if y_p < -self.grid_range:
            y_p = -self.grid_range
        NUMBER_OF_CELLS = (self.grid_range * 2) **2
        maxx = int(math.ceil(x_p / 10.0)) * 10
        maxy = int(math.ceil(y_p / 10.0)) * 10
        maxx += self.grid_range
        maxx = (maxx/self.cell_size) -1
        maxy += -self.grid_range
        maxy = (maxy/-self.cell_size) 
        
        #print(int(maxy), int(maxx))
        return zones[int(maxy)][int(maxx)]



    def update_probability(self, particles, zones):
        NUMBER_OF_PARTICLES = 1000
        count = 0
        for particle in particles: 
            zone_number = self.indexToCell(particle.x_p, particle.y_p, zones)
            zone_number[3] += 1/NUMBER_OF_PARTICLES
            count +=1
            #print(count)
        
        return zones
    
    def normalize_probability(self, zones):
        denominator = 0
        NUMBER_OF_ZONES = (self.grid_range/10)**2
        for zone in zones: 
            for cell in zone:
                denominator += cell[3]
        for zone in zones: 
            for cell in zone:
                cell[3] = cell[3]/denominator
        return zones

    def devalue_probability(self, zones):
        #print(zones)
        for zone in zones: 
            for cell in zone:
                if float(cell[3]) < .1: 
                    cell[3] += -.005
                elif float(cell[3]) < .5: 
                    cell[3] += -.01
                elif float(cell[3]) < 1: 
                    cell[3] += -.05
                if float(cell[3]) < 0: 
                    cell[3] = 0
        return zones

    def main_zone_function(self, particles, zones):
        self.devalue_probability(zones)
        #print("is anything happening")
        
        self.update_probability(particles, zones)
        
        self.normalize_probability(zones)
        #print("did i work")
        return zones

class Histogram: 
    def __init__(self, boundary_size, cell_size, number_of_sharks):
        self.boundary_size = boundary_size
        self.cell_size = cell_size
        #length of half of one side of the boundary "square", so that the origin is in the middle
        #generally 150meters
        self.number_of_sharks = number_of_sharks

    def create_zones(self):
        habitat_zone = HabitatZones((self.boundary_size), self.cell_size)
        zones = habitat_zone.create_zones(self.cell_size)
        for zone in zones:
            for cell in zone: 
                cell[2] = 0
                #number of times the shark leaves
                cell[3] = 0
                #number of times that the shark stays
        return zones

    def indexToCell(self, x_shark, y_shark, zones):
        #round x_p and y_p up to the nearest 10
        if x_shark > self.boundary_size:
            x_shark = self.boundary_size
        if x_shark < -self.boundary_size:
            x_shark = -self.boundary_size
        if y_shark > self.boundary_size:
            y_shark = self.boundary_size
        if y_shark < -self.boundary_size:
            y_shark = -self.boundary_size
        NUMBER_OF_CELLS = (self.boundary_size * 2) **2
        maxx = int(math.ceil(x_shark / 10.0)) * 10
        maxy = int(math.ceil(y_shark / 10.0)) * 10
        maxx += self.boundary_size
        maxx = (maxx/self.cell_size) -1
        maxy += -self.boundary_size
        maxy = (maxy/-self.cell_size)
        
        return zones[int(maxy)][int(maxx)]

    def create_historical_sharks(self):
        #shark_number = 0
        #sharks = []
        
        test_robot = RobotSim(5.0, 5.0, 0, 0.1)
        #sharks.append(shark_s)
        #test_robot.setup("./data/shark_tracking_data_x.csv", "./data/shark_tracking_data_y.csv", [0,31])
        shark_testing_trajectories = test_robot.load_shark_testing_trajectories("./data/shark_tracking_data_x.csv", "./data/shark_tracking_data_y.csv")
        return shark_testing_trajectories


    
            
    def update_probabilities_historical(self, shark_testing_trajectory, zones):
        id_number = 0
        new_cell = []
        old_cell = []
        for x in range(0, self.number_of_sharks-1):
            trajectory = shark_testing_trajectory[x].traj_pts_array
            #print(trajectory)
            for x in range(len(trajectory)):
                if x <= (len(trajectory)-2):
                    #print(x, len(trajectory))
                    new_time = trajectory[x+1]
                    old_time = trajectory[x]
                    new_coordinates = [new_time.x, new_time.y]
                    #print(new_coordinates[0])
                    new_cell = self.indexToCell(int(new_coordinates[0]), int(new_coordinates[1]), zones)
                    old_coordinates = [old_time.x, old_time.y]
                    old_cell = self.indexToCell(old_coordinates[0], old_coordinates[1], zones)
                    if new_cell == old_cell:
                        old_cell[3] += 1
                    elif new_cell != old_cell:
                        old_cell[2] += 1

        return zones

    def update_probabilities_random(self, shark_testing_trajectory, zones):
        id_number = 0
        new_cell = []
        old_cell = []
        for x in range(0, self.number_of_sharks-1):
            trajectory = shark_testing_trajectory[x].traj_pts_array
            #print(trajectory)
            for x in range(len(trajectory)):
                if x <= (len(trajectory)-2):
                    #print(x, len(trajectory))
                    new_time = trajectory[x+1]
                    old_time = trajectory[x]
                    new_coordinates = [new_time.x, new_time.y]
                    #print(new_coordinates[0])
                    new_cell = self.indexToCell(int(new_coordinates[0]), int(new_coordinates[1]), zones)
                    old_coordinates = [old_time.x, old_time.y]
                    old_cell = self.indexToCell(old_coordinates[0], old_coordinates[1], zones)
                    if new_cell == old_cell:
                        old_cell[3] += 1
                    elif new_cell != old_cell:
                        old_cell[2] += 1

        return zones

    def normalize_probabilities(self, zones):
        total = 0
        graphing_zones = []
        #graphing_zones = [[0 for x in range(int(math.ceil(self.boundary_size * 2) / self.cell_size))] for y in range(int(math.ceil(self.boundary_size * 2) / self.cell_size))]
        new_cell = ''
        for zone in zones:
            for cell in zone:
                total = cell[2] + cell[3]
                del cell[0:3]
                #print(cell)
                if total != 0:
                    cell[0] = cell[0]/total
                
                cell[0] = round(cell[0],3)
                
        zones = np.reshape(zones, (len(zones), len(zones)))
        #print(graphing_zones)
        print(zones)
        #print(type(cell))
    
        return zones
    
    def plot_probabilities(self, zones):
        
        y_axis = []
        for y in range(len(zones)):
            y_axis.append(y)
        x_axis = []
        for x in range(len(zones)):
            x_axis.append(x)
        #print(zones)
        probabilities = np.array(zones, dtype = np.float)


        fig, ax = plt.subplots()
        im = ax.imshow(probabilities)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_axis)))
        ax.set_yticks(np.arange(len(y_axis)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(x_axis)
        ax.set_yticklabels(y_axis)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax.text(j, i, probabilities[i, j],
                            ha="center", va="center", color="w")

        ax.set_title("Probability of shark to stay in current cell")
        fig.tight_layout()
        plt.show()
        

    

def main_histogram_function():
    histogram = Histogram(40, 10, 10)
    zones = histogram.create_zones()
    shark_testing_trajectories = histogram.create_historical_sharks()
    zones = histogram.update_probabilities_historical(shark_testing_trajectories, zones)
    zones = histogram.normalize_probabilities(zones)
    histogram.plot_probabilities(zones)
            








def main(): 
    x_mean_over_time = []
    y_mean_over_time = []
    num_of_loops = 1
    num_of_inner_loops = 400
    final_time_list = []
    for i in range(num_of_loops):
        NUMBER_OF_PARTICLES = 1000
        #change this constant ^^ in the particle class too!
        particles = []
        #test_grapher = Live3DGraph()
        #test_grapher_shark = Figure()
        #shark's initial x, y, z, theta
        #test_shark = RobotSim(740, 280, -5, 0.1)
        #test_shark.setup("./data/shark_tracking_data_x.csv", "./data/shark_tracking_data_y.csv", [1,2])
        x_auv = 0
        y_auv = 0
        theta = 0 
        x_auv_2 = -10
        y_auv_2 = 10
        theta_2 = 0
        # for now, since the auv is stationary, we can just set it like this
        #auv_pos = Motion_plan_state(x_auv, y_auv, theta)
        # example of how to get the shark x position and y position
        """
        shark_list = test_shark.live_graph.shark_array
        shark = test_shark.live_graph.shark_array[0]
        initial_x_shark = shark.x_pos_array[0]
        initial_y_shark = shark.y_pos_array[0]
        """
        shark_1 = Shark(5, 5)
        shark_2 = Shark(20, 10)
        test_list_shark1 = []
        test_list_shark2 = []
        

        test_particle = ParticleFilter(theta, x_auv ,y_auv, x_auv_2, y_auv_2, theta_2)
        final_new_shark_coordinate_x = []
        final_new_shark_coordinate_y = []
        actual_shark_coordinate_x = []
        actual_shark_coordinate_y = []
        """
        shark_state_dict = test_shark.get_all_sharks_state()
        has_new_data = test_shark.get_all_sharks_sensor_measurements(shark_state_dict, auv_pos)
        test_shark.shark_sensor_data_dict[1]
        test_particle.x_shark = test_shark.shark_sensor_data_dict[1].x
        test_particle.y_shark = test_shark.shark_sensor_data_dict[1].y
        final_new_shark_coordinate_x.append(test_particle.x_shark)
        final_new_shark_coordinate_y.append(test_particle.y_shark)
        """
        
        #CREATE HABITAT ZONES, measurements in meters
        grid_range = 150
        cell_size = 10 
        habitat_grid = HabitatZones(grid_range, cell_size)
        zones = habitat_grid.create_zones(cell_size)
        
        
        for x in range(0, NUMBER_OF_PARTICLES):
            new_particle = Particle(shark_1.x_shark, shark_1.y_shark)
            particles.append(new_particle)
        
        test_particle.create_and_update(particles)
        
        shark1_auv_alpha = test_particle.auv_to_alpha(shark_1.x_shark, shark_1.y_shark)
        shark1_auv_alpha_2 = test_particle.auv_to_alpha_2(shark_1.x_shark, shark_1.y_shark)
        shark1_auv_range = test_particle.range_auv(shark_1.x_shark, shark_1.y_shark)
        shark1_auv_range_2 = test_particle.range_auv_2(shark_1.x_shark, shark_1.y_shark)
        
        shark2_auv_alpha = test_particle.auv_to_alpha(shark_2.x_shark, shark_2.y_shark)
        shark2_auv_alpha_2 = test_particle.auv_to_alpha_2(shark_2.x_shark, shark_2.y_shark)
        shark2_auv_range = test_particle.range_auv(shark_2.x_shark, shark_2.y_shark)
        shark2_auv_range_2 = test_particle.range_auv_2(shark_2.x_shark, shark_2.y_shark)

        shark1_normalized_weights = test_particle.update_weights(particles, shark1_auv_alpha, shark1_auv_range, shark1_auv_alpha_2, shark1_auv_range_2)
        shark2_normalized_weights = test_particle.update_weights(particles, shark2_auv_alpha, shark2_auv_range, shark2_auv_alpha_2, shark2_auv_range_2)
        
        particles = test_particle.final_weight_correct(particles, shark1_normalized_weights, shark2_normalized_weights)
        
        #update habitat zone
        zones = habitat_grid.main_zone_function(particles, zones)
        
        
        particle_coordinates = test_particle.particle_coordinates(particles)
        loops = 0
        sim_time = 0.0
        sim_time_list = []
        index_number_of_particles = 0
        sim_time_list.append(sim_time)
        
        for j in range(num_of_inner_loops):
            
            time.sleep(.1)
            
                #print("x:", particle.x_p, " y:", particle.y_p, " velocity:", particle.v_p, " theta:", particle.theta_p, " weight:", particle.weight_p)
            #print("updated particles after", .1, "seconds of random movement")
            # update the shark position (do this in your main loop)
            for particle in particles: 
                particle.update_particle(.1)


            shark_1.update_shark(.1)
            shark_2.update_shark(.1)


            """
            for shark in shark_list:
                test_shark.live_graph.update_shark_location(shark, sim_time)
            
            shark_state_dict = test_shark.get_all_sharks_state()
            
            #print("==================")
            #print("All the Shark States [x, y, ..., time_stamp]: " + str(shark_state_dict))

            has_new_data = test_shark.get_all_sharks_sensor_measurements(shark_state_dict, auv_pos)

            if has_new_data == True:
                print("======NEW DATA=======")
                print("All The Shark Sensor Measurements [range, bearing]: " +\
                    str(test_shark.shark_sensor_data_dict))
            """
            shark1_auv_alpha = test_particle.auv_to_alpha(shark_1.x_shark, shark_1.y_shark)
            shark1_auv_alpha_2 = test_particle.auv_to_alpha_2(shark_1.x_shark, shark_1.y_shark)
            shark1_auv_range = test_particle.range_auv(shark_1.x_shark, shark_1.y_shark)
            shark1_auv_range_2 = test_particle.range_auv_2(shark_1.x_shark, shark_1.y_shark)

            shark2_auv_alpha = test_particle.auv_to_alpha(shark_2.x_shark, shark_2.y_shark)
            shark2_auv_alpha_2 = test_particle.auv_to_alpha_2(shark_2.x_shark, shark_2.y_shark)
            shark2_auv_range = test_particle.range_auv(shark_2.x_shark, shark_2.y_shark)
            shark2_auv_range_2 = test_particle.range_auv_2(shark_2.x_shark, shark_2.y_shark)

            shark1_normalized_weights = test_particle.update_weights(particles, shark1_auv_alpha, shark1_auv_range, shark1_auv_alpha_2, shark1_auv_range_2)
            shark2_normalized_weights = test_particle.update_weights(particles, shark2_auv_alpha, shark2_auv_range, shark2_auv_alpha_2, shark2_auv_range_2)
            
            particles = test_particle.final_weight_correct(particles, shark1_normalized_weights, shark2_normalized_weights)

            zones = habitat_grid.main_zone_function(particles, zones)
            print(loops)

            if loops%20 == 0:
                print("==============================")
                test_list_shark1 = shark_1.update_shark(.1)
                shark_1.x_shark = test_list_shark1[0]
                shark_1.y_shark = test_list_shark1[1]
                print("new shark 1 coordinates", test_list_shark1)
                test_list_shark2 = shark_2.update_shark(.1)
                shark_2.x_shark = test_list_shark2[0]
                shark_2.y_shark = test_list_shark2[1]
                print("new shark 2 coordinates", test_list_shark2)
                test_list = test_particle.update_auv(1)
                test_particle.x_auv = test_list[0]
                test_particle.y_auv = test_list[1]
                print("auv coordinates", test_list)
                test_list_2 = test_particle.update_auv_2(1)
                test_particle.x_auv_2 = test_list_2[0]
                test_particle.y_auv_2 = test_list_2[1]
                print("auv 2 coordinates ", test_list_2)

                print("---------------------------------")
                
            #print("mean of all particles (x, y): ", xy_mean)
            #x_mean_over_time.append(xy_mean[0])
            #y_mean_over_time.append(xy_mean[1])
            #final_new_shark_coordinate_x.append(test_particle.x_shark)
            #final_new_shark_coordinate_y.append(test_particle.y_shark)
            sim_time_list.append(sim_time)
            #test_particle.x_shark = test_shark.shark_sensor_data_dict[1].x
            #test_particle.y_shark = test_shark.shark_sensor_data_dict[1].y
            sim_time += 0.1
            loops += 1
            #print(loops)
            particle_coordinates = test_particle.particle_coordinates(particles)
            #print("+++++++++++++++++++++++++++++++")

            """
            test_shark.live_graph.plot_particles(particle_coordinates, final_new_shark_coordinate_x, final_new_shark_coordinate_y, actual_shark_coordinate_x, actual_shark_coordinate_y)
            plt.draw()
            plt.pause(.1)
            test_shark.live_graph.ax.clear()
            
            

        
        
        plt.close()
       # print(len(x_mean_over_time))
        #print(len(final_new_shark_coordinate_x))
        range_list = test_grapher_shark.range_plotter(x_mean_over_time, y_mean_over_time, final_new_shark_coordinate_x, final_new_shark_coordinate_y, sim_time_list)
        print("range list")
        print(range_list)
        
        time_list = test_grapher_shark.range_list_function(range_list, sim_time_list)
        final_time_list.append(time_list)
        print("final time list")
        print(time_list)
        plt.show()
        """

if __name__ == "__main__":
    main_shark_traj_function()