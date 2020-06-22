from drone_simulation_functions import *
import numpy as np

# physics
g = 9.81                                              # m/s^2
rho = 1.225                                           # kg/m^3

# drone parameters
max_thrust = 40                                       # N
max_rot_rate = 1                                      # rad/s (maximum rotational rate around z-axis)
mass = 1   
S = [0.01,0.01,0.03]                                  # [yz-plane,xz_plane,xy-plane] in m^2
CD = np.array([1,1,1])                                # CD of [yz-plane,xz_plane,xy-plane]
K = 0.5                                               # linear propeller drag coefficient
J = np.array([[1.1,0,0],[0,1,0],[0,0,1.7]])*10**(-3)  # Mass moment of inertia matrix
Jr = np.array([[0,0,0],[0,0,0],[0,0,1]])*10**(-6)     # Mass moment of inertia matrix




def drone_model(point, drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt):
    drone_rotation_matrix = rotation_matrix(drone_angles, drone=True)
    
    R = drone_rotation_matrix
    f1, f2, f3, f4 = max_thrust/4,max_thrust/4,max_thrust/4,max_thrust/4
    F = throttle*np.array([0,0,-(f1+f2+f3+f4)])
    D = -CD*0.5*rho*drone_velocity*abs(drone_velocity)*S-K*drone_velocity*mass
    G = np.array([0,0,g*mass])
    drone_acceleration = (np.dot(R,F)+D+G)/mass
    drone_velocity = drone_velocity+drone_acceleration*dt
    drone_position = drone_position + drone_velocity*dt
    
    
    
    drone_angles[2] = normalize_angle(drone_angles[2])
    right_direction = point-drone_position    
    
    point_angle = np.arctan2(right_direction[1],right_direction[0]) - drone_angles[2]
    point_angle = normalize_angle(point_angle)
    
    damping = 1
    if point_angle > 0:
        drone_angles[2] += max_rot_rate*dt*damping
    if point_angle < 0:
        drone_angles[2] -= max_rot_rate*dt*damping
    
    
    
    horizontal_distance_to_point = np.sqrt(right_direction[0]**2+right_direction[1]**2)
    drone_angles[1] = max(-0.5,-horizontal_distance_to_point*0.01+0.001)
    

    vertical_distance_to_point = point[2]-drone_position[2]
    
    throttle = mass*g/max_thrust
    if vertical_distance_to_point > 0.01 or vertical_distance_to_point < -0.01:
        throttle -= vertical_distance_to_point*0.01
    throttle = min(1,throttle)
    
    
    next_point = False
    distance_to_point = np.linalg.norm(right_direction)
    if distance_to_point < 0.1:
        next_point = True
    
    
    return next_point, drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt




def manual_drone_model(drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt):
    drone_rotation_matrix = rotation_matrix(drone_angles, drone=True)
    
    R = drone_rotation_matrix
    f1, f2, f3, f4 = max_thrust/4,max_thrust/4,max_thrust/4,max_thrust/4
    F = throttle*np.array([0,0,-(f1+f2+f3+f4)])
    D = -CD*0.5*rho*drone_velocity*abs(drone_velocity)*S-K*drone_velocity*mass
    G = np.array([0,0,g*mass])
    drone_acceleration = (np.dot(R,F)+D+G)/mass
    drone_velocity = drone_velocity+drone_acceleration*dt
    drone_position = drone_position + drone_velocity*dt
    
    next_point = False
    return drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt







