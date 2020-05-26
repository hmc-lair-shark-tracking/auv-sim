# add all the stuff we gotta import
import math
import decimal
import time
import numpy as np
import random
from numpy import random
import matplotlib.pyplot as plt
from numpy.random import uniform 
from numpy.random import randn
#from apscheduler.scheduler import Scheduler
import threading
from robotSim import RobotSim
from live3DGraph import Live3DGraph

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


class particleFilter:
    # 2 sets of initial data- shark's initial position and velocity, and position of AUV 
    # output- estimates the sharks position and velocity
    def __init__(self, init_x_shark, init_y_shark, init_theta, init_x_auv, init_y_auv):
        "how you create an object out of the particle Filter class i.e. "
        self.x_shark = init_x_shark
        self.y_shark = init_y_shark
        self.theta = init_theta
        self.x_auv = init_x_auv
        self.y_auv = init_y_auv
    
    def createParticles(self):
        L = 150
        N = 3

        list_of_particles = []
        #generate random x, y such that it follows circle formula
        coordinate_of_particle = []
        x_p = 0 
        y_p = 0 
        v_p = 0 
        theta_p = 0
        count = 0
        dict = {}
        while count <= N-1:
            x_p = random.uniform(-L, L)
            y_p= random.uniform(-L, L)
            weight_p = (1/N)
            if math.sqrt(x_p**2 + y_p**2) <= L:
                count+= 1
                v_p = random.uniform(0,5)
                theta_p = random.uniform(-math.pi, math.pi) 
                coordinate_of_particle.append([x_p, y_p, v_p, theta_p, weight_p])
        print("create particles")
        print(coordinate_of_particle)
        for x in range(count):
            for y in coordinate_of_particle:
                dict[x] = y
        #print("dict")
        #print(dict)
        return coordinate_of_particle
        #returns a dictionary w count and the particle coordinates and its weight

    def updateParticles(self, s_list, dt):
        sigma_v = 0.3 
        sigma_0 = math.pi/2
        for p in s_list:
            #change the v of particle and account for noise
            p[2] = int(p[2]) + random.uniform(0, sigma_v)
            p[3] = int(p[3]) + random.uniform(0, sigma_0)
                #print("new v, theta")
                #print(p[2], p[3])
                #print("initial ")
                #print(p[0])
            p[0] += p[2] * math.cos(p[3]) * dt 
                #print(p[0])
            p[1] += p[2] * math.sin(p[3]) * dt
        print("update")
        print(s_list)
        return s_list


    def predict(self, n_list, dt):
        list_alpha = []
        # can consider ignoring this
        #print(s_list)
        for p in n_list:
            range_of_particles = math.sqrt((int(self.y_auv)- p[1])**2 + (int(self.x_auv)-p[0])**2)
            alpha = math.atan2((int(self.y_auv) - p[1]), (int(self.x_auv) - p[0])) - self.theta
            k = ["range",range_of_particles,"alpha ", alpha]
            list_alpha.append(k)
        #print(" list of alphas")
        #print(list_alpha)
        return list_alpha
        #prints range and alpha of particles    
        #print(len(list_alpha))

    def auv_to_alpha(self):
        list_of_real_alpha = []
        real_alpha = math.atan2((int(self.y_auv)) - (int(self.y_shark)), (int(self.x_auv)) - (int(self.x_shark))) - self.theta
        real_alpha = angle_wrap(real_alpha)
        list_of_real_alpha1 = [real_alpha, -real_alpha]
        list_of_real_alpha.append(list_of_real_alpha1)
        #print("list of real alpha [0]")
        #print(list_of_real_alpha[0])
        #print(list_of_real_alpha[0][0])
        #print("list of alpha")
        #print(list_of_real_alpha)
        return list_of_real_alpha
    
    def lotek_Angle(self, list1):
       # converts a list of alpha from particles from radians to Lotek angle units for random particles
        
        #list1 = [[-1, 1], [-2, 2]]
        #print("list1")
        #print(list1)
        list_of_lotek = []
        for k in list1:
            r = (-(10 ** -6) * (float(k[3]) **3)) + 2 * (10**-5 * (float(k[3])) **2) + 0.0947 * int(k[3]) - 0.2757
            list_of_lotek.append(r)
        #print(list_of_lotek)
        return list_of_lotek

    def weight(self, lotek_angle, particles_range_alpha):
        #print(lotek_angle)
        sigma_alpha = 1 
        weights_list = []
        #print("beg of particles list")
        #print(particles_range_alpha)
        #print("end of particles list")
        #count_loop = 0 
        #print("beg of particles")
        #print(particles_range_alpha)
        #print("end of particles")
        for a in particles_range_alpha:
            for k in lotek_angle:
                #print("k")
                #print(k)
                #print("k[0] ")
                #print(k[0])
            #print(count_loop)
            #print(type(abs(a[3])))
                if a >= 0:
                    function = .001 + ((1/math.sqrt(2*math.pi*sigma_alpha))*(math.e**(((-abs(a-k[0]))**2)))/(2*(sigma_alpha)**2))
                    weights_list.append(function)
                elif a < 0: 
                    function = .001 + ((1/math.sqrt(2*math.pi*sigma_alpha))*(math.e**(((-abs(a-k[1]))**2)))/(2*(sigma_alpha)**2))
                    weights_list.append(function)
        # @print("list of weights")
        #print(weights_list)
        return weights_list
        # there will be a lot of random alpha values... 
        # so now we gotta compare the alpha of the lotex so the real_alpha w the alphas of the particles to change the alphas of the particles
        #insert crazy equation

    def normalize(self, weights_list):
        newlist = []
        denominator= sum(weights_list)
        for weight in weights_list:
            weight1 = (1/ denominator) * weight
            newlist.append(weight1)
        #print("new weight hopefully")
        #print(newlist)
        return newlist

    def correct(self, normalize_list, old_coordinates):
        list_of_coordinates = []
        list_of_new_particles = []
        new_particles = []
        copy = []
        count = -1
        for k in normalize_list:
            if k < 0.25:
                count += 1
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                #print("count, ", count, "x, y", old_coordinates[count][:2] )
                #print(list_of_new_particles)
            elif k < 0.50:
                count += 1 
                copy = old_coordinates[count][:4]
                copy.append(k)
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                #print(list_of_new_particles)
            elif k < 0.75:
                count += 1
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)

                #print(list_of_new_particles)
            elif k < 1.0:
                count += 1
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                list_of_new_particles.append(copy)
                #print(list_of_new_particles)
            else:
                print("something is not right with the weights")
        #print(list_of_new_particles)

        for n in range(len(normalize_list)): 
            x = random.choice(len(list_of_new_particles))
            new_particles.append(list_of_new_particles[x])
        print("new particles")
        print(new_particles)
        return new_particles

 
def main():
    test_particle = particleFilter(10, 10 , 30, 20 ,20)
    #test_grapher = Live3DGraph()
    coordinate_of_particle = test_particle.createParticles()
    s_list = test_particle.updateParticles(coordinate_of_particle, 1)
    list_alpha = test_particle.predict(coordinate_of_particle, 1)
    list_of_real_alpha = test_particle.auv_to_alpha()
    list_of_lotek = test_particle.lotek_Angle(list_alpha)
    weights_list = test_particle.weight(list_of_real_alpha, list_of_lotek)
    newlist = test_particle.normalize(weights_list)
    new_particles = test_particle.correct(newlist, s_list)

    while True:
        time.sleep(2.0)
        s_list = test_particle.updateParticles(new_particles, 1)
        list_alpha= test_particle.predict(new_particles, 1)
        list_of_lotek = test_particle.lotek_Angle(list_alpha)
        test_particle.weight(list_of_real_alpha, list_of_lotek)
        newlist= test_particle.normalize(weights_list)
        new_particles = test_particle.correct(newlist, s_list)
        #particle_array = test_particle.correct(newlist, s_list)
        """test_grapher.plot_particles(particle_array)
        
        plt.draw()
        plt.pause(0.5)
        test_grapher.ax.clear()"""
    














    
