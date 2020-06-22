import numpy as np
import matplotlib.pyplot as plt
from drone_simulation_functions import rotation_matrix

# physics
g = 9.81
p = 101325
T = 300
R = 287
rho = p/(R*T)
V = 10
l = 0.03
mu = 21*10**(-6)
Re = rho*V*l/mu

# drone parameters
thrust_over_weight = 4
mass = 1.325
thrust = thrust_over_weight*mass*9.81
K = 0.5

"""
Reynolds number is in order of 10^4
==> All CD's are 1 for cylinders and above Re
arms are at 45 degrees
"""


# angles of the drone
angles = np.array([0,0.129,0]) # rotation around x, y and z axis respectively

# velocities wrt to the drone coordinate frame
V_earth = np.array([10,0,0])                                                 # wrt earth coordinate frame, vx,vy and vz respectively
V_drone = np.dot(rotation_matrix(angles), V_earth)        #wrt to drone coordinate frame
Vx = V_drone[0]
Vy = V_drone[1]
Vz = V_drone[2]



# if you want it for many velocities
Vx = []
Vy = []
Vz = []

for i in np.linspace(0,50,1000):
    V_drone = np.array([i,0,0])
    Vx.append(V_drone[0])
    Vy.append(V_drone[1])
    Vz.append(V_drone[2])

Vx = np.array(Vx)
Vy = np.array(Vy)
Vz = np.array(Vz)



# all in cm, arms start outside the body
body_length = 12
body_radius = 8/2
arm_length = 15.24
arm_radius = 2.858/2
motor_height = 2.1
motor_radius = 2.8/2

area_front = 10**(-4)*(np.pi*body_radius**2 + 4*arm_length*arm_radius*2/np.sqrt(2) + 4*motor_height*motor_radius*2)
area_side = 10**(-4)*(body_radius*2*body_length + 4*arm_length*arm_radius*2/np.sqrt(2) + 4*motor_height*motor_radius*2)
area_top = 10**(-4)*(np.pi*body_radius**2 + 4*arm_length*arm_radius*2 + 4*np.pi*motor_radius**2)

drag_front = (0.5*rho*Vx**2*area_front + K*Vx*mass)
drag_side = (0.5*rho*Vy**2*area_side + K*Vy*mass)
drag_top = (0.5*rho*Vz**2*area_top + K*Vz*mass)

total_drag = np.linalg.norm([drag_front,drag_side,drag_top])

horizontal_thrust = total_drag     # in N
vertical_thrust = mass*g               # in N
total_thrust = np.linalg.norm([horizontal_thrust,vertical_thrust])
total_thrust = []
    
total_achievable_thust = thrust_over_weight*mass*g

plt.plot(Vx,drag_front)
for i in range(1000):
        total_thrust.append(np.linalg.norm([drag_front[i],vertical_thrust]))
plt.plot(Vx,total_thrust)

plt.ylabel('Force (N)')
plt.xlabel('Velocity (m/s)')
plt.legend(["Drag", "Thrust"])

plt.show()
#print('the drone can achieve ',total_achievable_thust,'. The given velocity requires ',total_thrust,' of thrust.')





