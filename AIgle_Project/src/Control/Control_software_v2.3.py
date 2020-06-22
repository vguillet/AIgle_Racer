# Control software for the AIgle drone
# by Weronika and Yestin
# based on:
# Large angle control, from 'Trajectory Generation and Control for Quadrotors'
# By Daniel Mellinger

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drag(rc_d):

    m = 1                   # [kg]
    rho = 1.225             # [kg/m^3]
    S = [0.01,0.01,0.03]    # surface of [yz-plane,xz_plane,xy-plane] [m^2]
    CD = np.array([1,1,1])  # CD of [yz-plane,xz_plane,xy-plane]
    Kd = 0.5                # linear propeller drag coefficient
    
    # returns the drag acceleration in [m/s2]
    return -CD*0.5*rho*rc_d*abs(rc_d)*S/m - Kd*rc_d   
def pltt():
    plt.figure()
    plt.plot(eplist[:k])
    plt.plot(poserr*np.ones((k,1)))
    plt.plot(epabslis)
def plt3(scen):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if scen==0:
        ax.set_xlim3d(-0.5,3.5)
        ax.set_ylim3d(-3.5,3.5)
    if scen==1:
        ax.set_ylim3d(1,4)
    ax.set_zlim3d(0.5,1.5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    for i in range(len(navlist)-1):
        ax.plot(poslist[navlist[i]:navlist[i+1],0],poslist[navlist[i]:navlist[i+1],1],poslist[navlist[i]:navlist[i+1],2])
        ax.scatter(nav[i][0],nav[i][1],nav[i][2])
def plti():
    plt.figure()
    plt.plot(ulis[:k,1:4])
def plta():
    plt.figure()
    plt.plot(np.rad2deg(angleslist[:k]))
    
def pltvelerr():
    plt.figure()
    plt.plot(evlist[:k])
    
def pltvelresponse():
    plt.figure()
    plt.plot(np.arange(0,k,1)*0.001,(vellist[:k]-evlist[:k])[:,0], label='Velocity step input')
    plt.plot(np.arange(0,k,1)*0.001,(vellist[:k])[:,0], label='System response')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
#---------------------------------- Constants ---------------------------------------#

g = 9.81        # [m/s]
m = 1.3           # [kg]
ixx = 1.1E-3    # [kg/m2]
iyy = 1.01E-3   # [kg/m2]
izz = 1.71E-3   # [kg/m2]
dt = 0.001      # time step [s]
maxthrust=4*1.3*9.81
maxmom=2.2
maxmomy=0.35
torpm=456.6/2869
scen=1  #scenarios. 0 is half circle, 1 is straight line, 2 is  disturbance



for blah in range(5):

    if blah == 0:
        val = 1
    elif blah == 1:
        val = 2
    elif blah == 2:
        val = 3
    elif blah == 3:
        val = 4
    elif blah == 4:
        val = 5
    
    
    Kp = 0.1*np.array([[1,0,0],
                          [0,1,0],
                          [0,0,1]])        # position gain matrix, positive definite
    Kv = 6*np.array([[1,0,0],
                          [0,1,0],
                          [0,0,1]])        # velocity gain matrix, positive definite
    Ka = val*np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])        # acceleration gain matrix, diagonal
    Kr = 30*0.08*np.array([[2,0,0],
                           [0,1,0],
                           [0,0,2]])        # rotation gain matrix, diagonal
    Kw = 60*0.008*np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])        # angular rate gain matrix, diagonal



    #------------------------------------ Input -----------------------------------------#
    
    phi = np.deg2rad(0)                 # initial roll angle in world frame [rad]
    theta = np.deg2rad(0)              # initial pitch angle in world frame [rad]
    psi = np.deg2rad(0)                 # initial yaw angle in world frame [rad]
    #rc = np.array([-0.19,-0.21,0.2])    # initial location x,y,z in world frame [m]
    rc=np.array([-0.8,3,1])
    rc_d = np.array([0.01,0.001,0.001])       # initial speed x_dot, y_dot, z_dot in world frame [m/s]
    ac_prop = rc_d*0.5                  # initial acceleration from the propellers x_dotdot, y_dotdot, z_dotdot [m/s2]
    angrat = np.array([np.deg2rad(0),np.deg2rad(0),np.deg2rad(0)])  # initial angular rate [rad/s]
    
    #----------------------------------- Gains ------------------------------------------#
    
        
    Krr=0.8
    poserr=0.3
    velerr=0.01
    acerr=0.1
    
    #--------------------------------- Navigation Input ---------------------------------#
    steps=5
    steps=np.linspace(0,np.pi/2,steps)
    nav=3*np.array([np.sin(steps),np.cos(steps)]).T
    nav=np.hstack((nav,np.ones((1,len(nav))).T))
    nav=np.hstack((nav,rc_d[0]*np.ones((1,len(nav))).T))
    if scen==1 or scen==2:
        nav=np.array([[9,3,1,2],
                      [18,3,1,2],
                      [27,3,1,2]])
    
    #nav=np.array([[10,3,1,5],
    #              [5,5,5,5]])
    
    #nav = np.array([[0.05,-0.17,0.21,2.0],
    #                [0.15,-0.14,0.22,2.5],
    #                [0.29,-0.12,0.23,3.0],
    #                [0.50,-0.01,0.24,3.5],
    #                [0.73, 0.10,0.25,4.0],
    #                [1.00, 0.17,0.26,4.0],
    #                [1.31, 0.27,0.27,4.2],
    #                [1.62, 0.46,0.28,4.5],
    #                [1.92, 0.70,0.29,4.8],
    #                [2.20, 0.97,0.30,5.0],
    #                [2.49, 1.32,0.31,5.1]])
    
    #----------------------------------- Plotting ---------------------------------------#
    #iterations
    maxtrac=220
    maxvel=30
    maxat=30
    
    poslist = np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    eplist=np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    evlist=np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    aclist=np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    ulis=np.zeros((maxtrac*maxvel*maxat*len(nav),4))
    angleslist=np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    vellist=np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    disturb = np.zeros((maxtrac*maxvel*maxat*len(nav),3))
    if scen==2:
        disturb[3][1] = -10000
    epabslis=[]
    k=1
    poslist[0,:]=rc
    navlist=[]
    u_des=np.array([13,0,0,0])
    #---------------------------------- Loop --------------------------------------------#
    
    # Initial errors
    ep_abs = 10
    ev_abs = 10
    ea_abs = 10
    
    d=0
    e=0
    f=0
    j = 0
    break_p=0
    # Run the loop to reach each point from navigation




    for i in range(len(nav)):
        #poserr=0.1+.1*i/len(nav)
        #--------------------------------- Trajectory control loop -----------------------------------#
        
        # Keep going until the position error gets very small, then move on to the next point  
        rt = nav[i][0:3]                                          # desired position
        ep = rc - rt                                              # position error
        ep_abs = np.linalg.norm(ep)                               # absolute value of the error
        navlist.append(k)
        if break_p > maxtrac or ep_abs>10:
            break
        break_p = 0
        while ep_abs > poserr:
            
            # Stop the loop after too many iterations
            break_p += 1
            if break_p > maxtrac or ep_abs>10:
                break
            
            # Position and velocity errors
            rt = nav[i][0:3]                                          # desired position
            ep = rc - rt                                              # position error
            ep_abs = np.linalg.norm(ep)                               # absolute value of the error
            traj = (-ep)/np.linalg.norm((-ep))                        # trajectory unit vector
            rt_d = nav[i][3]*traj                                     # desired speed
            rt_d[0] = nav[i][3]
            ev = rc_d - rt_d                                          # velocity error
            ev_abs = np.linalg.norm(ev)                               # absolute value of the error
            
            # If the velocity is right, then keep going until the point is reached 
            # or until velocity is not right anymore
            if ev_abs < velerr:
                ac_tot = np.array([0,0,-g]) + ac_prop + drag(rc_d)  + disturb[k]               # total drone acceleration
                rc += rc_d*dt + 0.5*(ac_tot)*(dt**2)                  # new position
                poslist[k,:]=rc
                eplist[k,:]=ep
                aclist[k,:]=ac_prop
                ulis[k,:]=u_des.T
                angleslist[k,:]=np.array([phi,theta,psi]).T
                vellist[k,:]=rc_d
                epabslis.append(ep_abs)
                d+=1
                k+=1                                    # appending position to list
                rc_d += ac_tot*dt                                     # new velocity  
                ev = rc_d - rt_d
                evlist[k,:]=ev
            # But if the velocity is not right, then change acceleration until it is
            
        #--------------------------------- Velocity control loop -------------------------------------#
            
        # Keep going until the velocity error is very small, then just keep this velocity 
        # and keep flying till the position error is very small
            
            break_v = 0
            while ev_abs > velerr:
                
                # Stop the loop after too many iteration or if the waypoint is reached
                break_v += 1
                if break_v > maxvel or ep_abs < poserr:
                    break
                
                # Current rotation matrix
                z_b =  ac_prop/np.linalg.norm(ac_prop)                   # z body axis
                x_c = np.array([float(np.cos(psi)),float(np.sin(psi)),0]) # heading vector
                cross1 = np.cross(z_b,x_c)                                # cross product
                y_b = cross1/np.linalg.norm(cross1)                       # y body axis
                x_b = np.cross(y_b,z_b)                                   # x body axis
                rot = (np.vstack((x_b,y_b,z_b))).T                        # current rotation matrix
                
                # Desired acceleration to reach this velocity
                rt_dd = Ka.dot(-ev)                                       # desired acceleration 
                z_w = np.array([0,0,1])                                   # z world axis     
                F_des = - Kp.dot(ep) - Kv.dot(ev) + m*rt_dd + m*g*z_w     # desired force
                ea = F_des/m - ac_prop                                    # acceleration error
                ea_abs = np.linalg.norm(ea)                               # absolute value of the error
                
                # If the thrust force is correct, then keep accelerating at the same rate until the 
                # desired velocity is reached and then maintain it till it gets to the waypoint
                
                if ea_abs < acerr:
                    ac_tot = np.array([0,0,-g]) + ac_prop + drag(rc_d) +disturb[k]               # total drone acceleration
                    rc += rc_d*dt + 0.5*(ac_tot)*(dt**2)                  # new position
                    poslist[k,:]=rc
                    eplist[k,:]=ep
                    aclist[k,:]=ac_prop
                    ulis[k,:]=u_des.T
                    angleslist[k,:]=np.array([phi,theta,psi]).T
                    vellist[k,:]=rc_d
                    epabslis.append(ep_abs)
                    e+=1
                    k+=1                                      # appending position to list
                    rc_d += ac_tot*dt                                     # new velocity 
                    ev = rc_d - rt_d
                    evlist[k,:]=ev
                # Position and velocity errors
                ep = rc - rt                                              # position error
                ep_abs = np.linalg.norm(ep)                               # absolute value of the error
                traj = (-ep)/np.linalg.norm((-ep))                        # trajectory unit vector
                rt_d = nav[i][3]*traj                                     # desired speed
                rt_d[0] = nav[i][3]
                ev = rc_d - rt_d                                          # velocity error
                ev_abs = np.linalg.norm(ev)                               # absolute velue of the error
    
                
        #----------------------------------- Attitude control loop -----------------------------------#
        
        # If the thrust force is not correct, then you gotta adjust throttle and attitude till it is and 
        # then you can keep flying at this attitude and velocity till waypoint is reached
        
                break_a = 0
                while ea_abs > acerr:
                    
                    # Stop the loop after too many iterations or if the waypoint is reached
                    break_a += 1
                    if break_a > maxat or ep_abs < poserr:
                        break
                    
                    # Position and velocity errors
                    ep = rc - rt                                              # position error
                    ep_abs = np.linalg.norm(ep)                               # absolute value of the error
                    traj = (-ep)/np.linalg.norm((-ep))                        # trajectory unit vector
                    rt_d = nav[i][3]*traj                                     # desired speed
                    rt_d[0] = nav[i][3]
                    ev = rc_d - rt_d                                          # velocity error
                    ev_abs = np.linalg.norm(ev)                               # absolute velue of the error
    
                    # Current rotation matrix
                    z_b =  ac_prop/np.linalg.norm(ac_prop)                   # z body axis
                    x_c = np.array([float(np.cos(psi)),float(np.sin(psi)),0]) # heading vector
                    cross1 = np.cross(z_b,x_c)                                # cross product
                    y_b = cross1/np.linalg.norm(cross1)                       # y body axis
                    x_b = np.cross(y_b,z_b)                                   # x body axis
                    rot = (np.vstack((x_b,y_b,z_b))).T                        # current rotation matrix
                    
                    # Desired rotation matrix
                    rt_dd = Ka.dot(-ev)                                       # desired acceleration 
                    z_w = np.array([0,0,1])                                   # z world axis     
                    F_des = - Kp.dot(ep) - Kv.dot(ev) + m*rt_dd + m*g*z_w     # desired force
                    ea = F_des/m - ac_prop                                    # acceleration error
                    ea_abs = np.linalg.norm(ea)                               # absolute value of the error
                    z_b_des =  F_des/np.linalg.norm(F_des)                   # desired z body axis
                    psi_des = np.arctan((rt[1]/(rt[0]+0.001)))                        # desired heading angle
                    x_c_des = np.array([np.cos(psi_des),np.sin(psi_des),0])   # desired heading vector
                    cross2 = np.cross(z_b_des,x_c_des)                        # cross product
                    y_b_des = cross2/np.linalg.norm(cross2)                   # desired y body axis
                    x_b_des = np.cross(y_b_des,z_b_des)                       # desired x body axis
                    rot_des = (np.vstack((x_b_des,y_b_des,z_b_des))).T        # desired rotation matrix
                    
                    # Calculating output
                    if F_des[2]<0:
                        F_des[2]=0
                    u1_des = np.linalg.norm(F_des[2])                             # desired throttle        
                    u1_des = min(u1_des,maxthrust)
                    u1_des = max(u1_des,5)
                    er_mat = (rot_des.T).dot(rot) - (rot.T).dot(rot_des)      # for calculating the rotation error
                    er = np.array([er_mat[2][1],er_mat[0][2],er_mat[1][0]])/2 # rotation error 
                    angrat_des = Krr*er                                       # desired angular rate                 
                    ew = angrat - angrat_des                                  # angular rate error
                    u234_des = - np.dot(Kr,er) - np.dot(Kw,ew)                # desired moments 
                    u234_des[0]=max(-maxmom,min(u234_des[0],maxmom))
                    u234_des[1]=max(-maxmom,min(u234_des[1],maxmom))
                    u234_des[2]=max(-maxmomy,min(u234_des[2],maxmomy))
                    u_des = np.vstack((u1_des,u234_des[0],u234_des[1],u234_des[2])) # output vector 
                    
                    # Achieved state
                    phi += (0.5*u_des[1]*dt**2 )/ixx             # new roll angle
                    theta += (0.5*u_des[2]*dt**2 )/iyy           # new pitch angle
                    psi += (0.5*u_des[3]*dt**2 )/izz             # new yaw angle
                    rot_real = np.array([[float(np.cos(psi)*np.cos(theta)), float(np.cos(psi)*np.sin(phi)*np.sin(theta)-np.cos(phi)*np.sin(psi)),float(np.sin(psi)*np.sin(phi)+np.sin(theta)*np.cos(phi)*np.cos(psi))],
                                         [float(np.cos(theta)*np.sin(psi)),float(np.cos(phi)*np.cos(psi)+np.sin(phi)*np.sin(psi)*np.sin(theta)),float(np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi))],
                                         [float(-np.sin(theta)),float(np.cos(theta)*np.sin(phi)),float(np.cos(phi)*np.cos(theta))]]) #rotation matrix from body to world frame
                    ac_prop = np.dot(rot_real,np.array([0,0,float(u_des[0])]))/m # acceleration from propellers
                    ea = F_des/m - ac_prop                                    # acceleration error
                    ea_abs = np.linalg.norm(ea)                               # absolute value of the error
                    ac_tot = np.array([0,0,-g]) + ac_prop+ drag(rc_d) + disturb[k]                        # total drone acceleration
                    rc += rc_d*dt + 0.5*(ac_tot)*(dt**2)                      # new position
                    poslist[k,:]=rc
                    eplist[k,:]=ep
                    aclist[k,:]=ac_prop
                    ulis[k,:]=u_des.T
                    angleslist[k,:]=np.array([phi,theta,psi]).T
                    vellist[k,:]=rc_d
                    epabslis.append(ep_abs)
                    f+=1
                    k+=1                                          # appending position to list
                    rc_d += ac_tot*dt                                         # new velocity
                    ev = rc_d - rt_d
                    evlist[k,:]=ev
                    angrat = np.array([float(u_des[1]/ixx),float(u_des[2]/iyy),float(u_des[3]/izz)])*dt # new angular rate
                    
    navlist.append(k)
    # Position plot, red point is starting location
    
    plt.plot(np.arange(0,k,1)*0.001,(vellist[:k])[:,0], label='System response for P gain of 6 and I gain of 0.2 and D gain of '+str(val))
    plt.xlim(-0.5,14.)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    
plt.plot(np.arange(0,k,1)*0.001,(vellist[:k]-evlist[:k])[:,0], label='Velocity step input')   
plt.legend(loc='lower right') 
plt.show()