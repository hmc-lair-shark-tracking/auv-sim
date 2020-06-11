import math
import decimal
import time
import numpy as np
import random
from copy import copy, deepcopy
from numpy import random

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
        def __init__(self):
            #set L (side length of square that the random particles are in) and N (number of particles)
            INITIAL_PARTICLE_RANGE = 150
            NUMBER_OF_PARTICLES = 1000
            #particle has 5 properties: x, y, velocity, theta, weight (starts at 1/N)
            self.x_p = random.uniform(-INITIAL_PARTICLE_RANGE, INITIAL_PARTICLE_RANGE)
            #self.y_p = random.uniform(-INITIAL_PARTICLE_RANGE, INITIAL_PARTICLE_RANGE)
            self.y_p = 0
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
            #self.y_p += self.v_p * math.sin(self.theta_p) * dt
            
            
        def calc_particle_alpha(self, x_auv, y_auv, theta_auv):
            """
                calculates the alpha value of a particle
            """
            particleAlpha = angle_wrap(math.atan2((y_auv + self.y_p), (self.x_p + -x_auv))) - theta_auv
            return particleAlpha

        def calc_particle_range(self, x_auv, y_auv):
            """
                calculates the range from the particle to the auv
            """
            particleRange = math.sqrt((y_auv - self.y_p)**2 + (x_auv - self.x_p)**2)
            return particleRange

        def weight(self, auv_alpha, particleAlpha, auv_range, particleRange):
            """
                calculates the weight according to alpha, then the weight according to range
                they are multiplied together to get the final weight
            """
            
            #alpha weight
            SIGMA_ALPHA = .05
            
            if particleAlpha > 0:
                function_alpha = .001 + (1/(SIGMA_ALPHA * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(float(particleAlpha) - float(auv_alpha[0]))**2)))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = angle_wrap(function_alpha)
            elif particleAlpha == 0:
                function_alpha = .001 + (1/(SIGMA_ALPHA * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(float(particleAlpha) - float(auv_alpha[0]))**2)))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = angle_wrap(function_alpha)
            else:
                function_alpha = .001 + (1/(SIGMA_ALPHA * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(float(particleAlpha) + float(auv_alpha[0]))**2)))/(2*(SIGMA_ALPHA)**2))))
                self.weight_p = angle_wrap(function_alpha)
            
            #range weight
            SIGMA_RANGE = 10
            function_weight =  .001 + (1/(SIGMA_RANGE * math.sqrt(2*math.pi))* (math.e**(((-((particleRange-auv_range[0])**2)))/(2*(SIGMA_RANGE)**2))))
            
            #multiply weights
            self.weight_p = function_weight * self.weight_p