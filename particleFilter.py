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

sigma_alpha = 10
N = 1000
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
        # list of coordinates of particles: [x_p, y_p, v_p, theta_p, weight_p]
        L = 150
        list_of_particles = []
        coordinate_of_particle = []
        x_p = 0 
        y_p = 0 
        v_p = 0 
        theta_p = 0
        count = 0
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
        return coordinate_of_particle


    def updateParticles(self, s_list, dt):
        #updated particle list [x_p, y_p, v_p, theta_p, weight_p]
        sigma_v = 5
        sigma_0 =  math.pi
        for p in s_list:
            #change the v of particle and account for noise
            p[2] = float(p[2]) + random.uniform(0, sigma_v)
            p[3] = float(p[3]) + random.uniform(0, sigma_0)
                #print("new v, theta")
            p[0] += p[2] * math.cos(p[3]) * dt 
                #print(p[0])
            p[1] += p[2] * math.sin(p[3]) * dt
        #print("update")
        #print(s_list)
        return s_list

    def updateShark(self, dt):
        v_x_shark = 0
        v_y_shark = .1
        self.x_shark = self.x_shark + (v_x_shark * dt)
        self.y_shark = self.y_shark + (v_y_shark * dt)
        return [self.x_shark, self.y_shark]

    def predict(self, n_list, dt):
        #predicts what the particles' alpha values are
        list_alpha = []
        #p[0] --x of particle
        #p[1] --y of the particle
        for p in n_list:
            range_of_particles = math.sqrt((float(self.y_auv)- p[1])**2 + (float(self.x_auv)-p[0])**2)
            alpha = angle_wrap(math.atan2((-self.y_auv + p[1]), (p[0] + -self.x_auv))) - self.theta
            k = ["range",range_of_particles,"alpha ", alpha]
            list_alpha.append(k)
        print(list_alpha)
        return list_alpha
        #prints range and alpha of particles    
        #print(len(list_alpha))

    def auv_to_alpha(self):
        #calculates auv's alpha from the shark
        list_of_real_alpha = []
        real_alpha = angle_wrap(math.atan2((-self.y_auv + self.y_shark), (self.x_shark- self.x_auv))) - self.theta
        list_of_real_alpha.append(real_alpha)
        list_of_real_alpha.append(-real_alpha)
        #print("list of alpha")
        #print(list_of_real_alpha)
        return list_of_real_alpha

    def range_auv(self):
        range = []
        range_value = math.sqrt((float(self.y_auv)- self.y_shark)**2 + (float(self.x_auv)-float(self.x_shark))**2)
        range.append(range_value)
        print("range of auv is")
        print(range)
        return range

    def lotek_Angle(self, list1):
       # converts a list of alpha from particles from radians to Lotek angle units for random particles
        list_of_lotek = []
        for k in list1:
            r = (-(10 ** -6) * (float(k[3]) **3)) + 2 * (10**-5 * (float(k[3])) **2) + 0.0947 * int(k[3]) - 0.2757
            list_of_lotek.append(r)
        #print(list_of_lotek)
        return list_of_lotek

    def weight_checker(self, k_value, auv_value):
        sigma_alpha = 0.5
        print("particle alpha:")
        print(k_value)
        print("auv alpha:")
        print(auv_value)
        function = 1/N + (1/(sigma_alpha * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(k_value-auv_value)**2)))/(2*(sigma_alpha)**2))))
        print("weight is:")
        print(function)

# write a function --> to check weights
    def weight(self, auv_alpha, particles_alpha):
        #sigma_alpha = 13.4585
        sigma_alpha = 0.5
        weights_list = []
        print("auv_alpha")
        print(auv_alpha)
        positive_auv_alpha = auv_alpha[1]
        negative_auv_alpha = auv_alpha[0]
        for k in particles_alpha:
            if k[3] > 0:
                function = .001 + (1/(sigma_alpha * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(k[3]-positive_auv_alpha)**2)))/(2*(sigma_alpha)**2))))
                #self.weight_checker(k[3], positive_auv_alpha)
                function = angle_wrap(function)
                weights_list.append(function)
            elif k[3] == 0:
                function = .001 + (1/(sigma_alpha * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(k[3]-positive_auv_alpha)**2)))/(2*(sigma_alpha)**2))))
                #self.weight_checker(k[3], positive_auv_alpha)
                function = angle_wrap(function)
                weights_list.append(function)
            else:
                function = .001 + (1/(sigma_alpha * math.sqrt(2*math.pi))* (math.e**(((-(angle_wrap(k[3]-negative_auv_alpha)**2)))/(2*(sigma_alpha)**2))))
                #self.weight_checker(k[3], negative_auv_alpha)
                function = angle_wrap(function)
                weights_list.append(function)
        print("list of weights through alpha")
        print(weights_list)
        return weights_list
    
    def weight_range(self, particles_alpha, range_auv):
        # maybe just uses the distance formula between the particles range and the auv range
        weight_range = []
        sum = 0
        #k[1] and r[1] is the range of the particles
        sigma_range = 10
        for k in particles_alpha:
            weight =  .001 + (1/(sigma_range * math.sqrt(2*math.pi))* (math.e**(((-((k[1]-range_auv[0])**2)))/(2*(sigma_range)**2))))
            weight_range.append(float(weight))
        print("weight of range list: ")
        print(weight_range)
        return weight_range

            
    def normalize(self, weights_list):
        newlist = []
        denominator= max(weights_list)
        for weight in weights_list:
            weight1 = (1/ denominator) * weight
            newlist.append(weight1)
        #print("new weight hopefully")
        #print(newlist)
        return newlist

    def combined_weights(self, weights_list, weight_range_list):
        new_weight_list = []
        weight_list = []
        count = -1
        for weight in weights_list:
            count += 1
            weight_final = weight * weight_range_list[count]
            weight_list.append(weight_final)
        new_weight_list = self.normalize(weight_list)
        print("trial new weight list")
        print(new_weight_list)
        return new_weight_list
        
    def correct(self, normalize_list, old_coordinates):
        list_of_coordinates = []
        list_of_new_particles = []
        new_particles = []
        copy = []
        count = -1
        for k in normalize_list:
            if k < 0.2:
                count += 1
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                
                #list_of_new_particles.append(copy)
                #print("count, ", count, "x, y", old_coordinates[count][:2] )
                #print(list_of_new_particles)
            elif k < 0.4:
                count += 1 
                copy = old_coordinates[count][:4]
                copy.append(k)
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                list_of_new_particles.append(copy)
                copy1 = old_coordinates[count][:4]
                copy1.append(k)
                list_of_new_particles.append(copy1)
                
                #print(list_of_new_particles)
            elif k < 0.6:
                count += 1
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                copy1 = old_coordinates[count][:4]
                copy1.append(k)
                list_of_new_particles.append(copy1)
                copy2 = old_coordinates[count][:4]
                copy2.append(k)
                list_of_new_particles.append(copy2)

                #print(list_of_new_particles)
            elif k < .8:
                count += 1
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                copy1 = old_coordinates[count][:4]
                copy1.append(k)
                list_of_new_particles.append(copy1)
                copy2 = old_coordinates[count][:4]
                copy2.append(k)
                list_of_new_particles.append(copy2)
                copy3 = old_coordinates[count][:4]
                copy3.append(k)
                list_of_new_particles.append(copy3)
                
                #print(list_of_new_particles)
            elif k <= 1.0:
                count += 1
                #print("count,", count, "x, y ", old_coordinates[count][:2])
                copy = old_coordinates[count][:4]
                copy.append(k)
                list_of_new_particles.append(copy)
                copy1 = old_coordinates[count][:4]
                copy1.append(k)
                list_of_new_particles.append(copy1)
                copy2 = old_coordinates[count][:4]
                copy2.append(k)
                list_of_new_particles.append(copy2)
                copy3 = old_coordinates[count][:4]
                copy3.append(k)
                list_of_new_particles.append(copy3)
                copy4 = old_coordinates[count][:4]
                copy4.append(k)
                list_of_new_particles.append(copy4)
            else:
                print("something is not right with the weights")
        #print('list of new particles')
        #print(list_of_new_particles)

        for n in range(len(normalize_list)): 
            x = random.choice(len(list_of_new_particles))
            new_particles.append(list_of_new_particles[x])
        print("new particles")
        print(new_particles)
        return new_particles

def main():

    # coordinates are x_shark, y_shark, theta, x_auv, y_auv
    test_grapher = Live3DGraph()
    test_particle = particleFilter(0, 0 , 0, 0 ,1)
    coordinate_of_particle = test_particle.createParticles()
    #coordinate_of_particle = [[1, 1 , 3, 1.4, 0.333333],[0, 0, 3, 1.4, 0.333333], [0, 2, 3, 1.4, 0.333333]]
    randomize_particles = test_particle.updateParticles(coordinate_of_particle, .1)
    auv_alpha = test_particle.auv_to_alpha()
    print("coordinates of particle")
    print(randomize_particles)
    particles_alpha = test_particle.predict(randomize_particles, 1)
    range_auv = test_particle.range_auv()
    weights_list = test_particle.weight(auv_alpha, particles_alpha)
    weight_range_list = test_particle.weight_range(particles_alpha, range_auv)
    final_weights = test_particle.combined_weights(weights_list, weight_range_list)
    new_particles = test_particle.correct(final_weights, randomize_particles)
    # new particles based on first weight
    while True:
        time.sleep(.1)
        auv_alpha = test_particle.auv_to_alpha()
        print("coordinates of particle")
        print(randomize_particles)
        particles_alpha = test_particle.predict(randomize_particles, 1)
        range_auv = test_particle.range_auv()
        weights_list = test_particle.weight(auv_alpha, particles_alpha)
        weight_range_list = test_particle.weight_range(particles_alpha, range_auv)
        final_weights = test_particle.combined_weights(weights_list, weight_range_list)
        new_particles = test_particle.correct(final_weights, randomize_particles)
        randomize_particles = test_particle.updateParticles(new_particles, .1)
        new_coordinates = test_particle.updateShark(1)
        
        #simulation stuff
        test_grapher.plot_particles(new_particles)
        
        plt.draw()
        plt.pause(0.5)
        test_grapher.ax.clear()

    
    #creates particles 



if __name__ == "__main__":
    main()