import math


def get_auv_trajectory(v,delta_t):
    traj_list = []
    t = 0
    x = 10
    y = 10
    for i in range(20):
        t = t + delta_t
        x = x + v*delta_t
        y = y
        theta = 0

        traj_list.insert(len(traj_list), [t,x,y,theta])

    for i in range(20):
        t = t + delta_t
        x = x
        y = y + v*delta_t
        theta = math.pi/2

        traj_list.insert(len(traj_list), [t,x,y,theta])     
   
    for i in range(20):
        t = t + delta_t
        x = x+ v*delta_t
        y = y 
        theta = math.pi

        traj_list.insert(len(traj_list), [t,x,y,theta]) 

    for i in range(20):
        t = t + delta_t
        x = x
        y = y + v*delta_t 
        theta = 3*(math.pi)/2

        traj_list.insert(len(traj_list), [t,x,y,theta]) 
    return traj_list
    
def main():
    v=10
    delta_t = 0.1
    get_auv_trajectory(v,delta_t)
if __name__ == "__main__":
    main()    