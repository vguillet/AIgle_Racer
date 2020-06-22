#------------------------------------ IMPORTS -----------------------------------------
from drone_simulation_functions import *
from drone_model import *
import numpy as np
from math import sin,cos
import pygame as pg
pg.init()

#------------------------------------ INITIAL VARIABLES -----------------------
# control mode
manual_control = True

# path (for now only visualization, no path finding)
waypoints = np.array([[0,0,0],[1,7,8],[8,1,3],[4,4,8],[1,3,7],[10,10,10]])        # waypoints drone will go to in order. First one is starting position
draw_points = False
draw_path = True                                
draw_gates = False

# starting values
drone_position = waypoints[0]
drone_velocity = np.array([0,0,0])
drone_angles = [0,0,0]
drone_rotation_matrix = np.identity(3)
throttle = 0


# colors and fonts
background_color             = color.black
path_color                   = color.white
begin_point_color            = color.green
waypoint_color               = color.yellow
end_point_color              = color.red
axis_color                   = color.lime
text_color                   = color.lightblue
drone_rotor_color            = color.white
drone_diagonal_color         = color.red
drone_facing_direction_color = color.blue
text_font = pg.font.SysFont('HELVETICA', 20, bold=True, italic=False)


# visuals
screensize = np.array([800,800])
origin = np.array([100,100])        # w.r.t. left upper corner of screen, should be np.array([screensize[0]/2,screensize[0]/2]) for MACHINE VISION to be centered
axis_length = 100                   # cm
angles = np.array([0.0,0.2,0.2])    # viewing angles of general coordinate frame, should be np.array([0,0,0]) for MACHINE VISION
drone_axis_length = 10              # cm
drone_size = [20,20,10]             # [x,y,z] dimensions (WITHOUT rotors) in cm. Only used for visualizing the drone
rotor_radius = 7                    # cm
point_size = 1
path_point_size = 0.1
scale = 10

# initiation variables
mouse_position = pg.mouse.get_pos()
path_points = np.array([0,0,0])                 # add values to this in the code using .append
waypoint_number = 0

#--------------------------------------- SET SCREEN ---------------------------
fullscreen = False # full screen does not seem to work accurately in pygame??

if fullscreen == True:
    scr = pg.display.set_mode((0, 0), pg.FULLSCREEN)
if fullscreen == False:
    scr = pg.display.set_mode((screensize[0],screensize[1]))

#------------------------------------- MAIN -----------------------------------
tprev = pg.time.get_ticks()*0.001
running = True
while running:
    time = pg.time.get_ticks()*0.001
    dt = time-tprev
    tprev = time
 
    #---------------------- screen manipulation with mouse --------------------
    events = pg.event.get()
    for event in events:
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 4:
                scale += 0.1*scale
                origin = origin + 0.1*(origin - np.array(pg.mouse.get_pos()))
            if event.button == 5:
                scale -= 0.1*scale
                origin = 0.9*origin+0.1*np.array(pg.mouse.get_pos())

    previous_mouse_position = mouse_position
    mouse_position = pg.mouse.get_pos()
    mouse_clicks = pg. mouse. get_pressed()
    if mouse_clicks[0] == 1:
        origin = origin+(np.array(mouse_position)-np.array(previous_mouse_position))
    if mouse_clicks[2] == 1:
        mouse_movement = np.array(mouse_position)-np.array(previous_mouse_position)
        angles[2] -= 0.01*mouse_movement[0]
        angles[1] += 0.01*mouse_movement[1]
        
    #------------------------ manual drone control ----------------------------
    
    if manual_control == True:
        throttle = min(1,max(drone_position[2]+0.2,0.1))
        drone_angles = [0,0,drone_angles[2]]
        for key in pressed_keys():
            if key == 'up':
                throttle = 1
            if key == 'down':
                throttle = 0
            if key == 'w':
                drone_angles[1] = -0.2
            if key == 's':
                drone_angles[1] = 0.2
            if key == 'd':
                drone_angles[0] = 0.2
            if key == 'a':
                drone_angles[0] = -0.2
            if key == 'q':
                drone_angles[2] -= 0.04
            if key == 'e':
                drone_angles[2] +=0.04
                
        drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt = manual_drone_model(drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt)

                
    #---------------- automatic drone control using drone model----------------

    if manual_control == False:
        point = waypoints[waypoint_number]
        next_point, drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt = drone_model(point, drone_position, drone_velocity, drone_angles, drone_rotation_matrix, throttle, dt)
        if next_point == True:
            waypoint_number += 1       
    
    #---------------------------- plot everything -----------------------------
    # make background black
    scr.fill(background_color)

    #-------------------------- plot waypoints --------------------------------
    # draw start point, end point and waypoints
    if draw_points == True:
        for point_number in range(len(waypoints)):
            if point_number == 0:
                pg.draw.circle(scr, begin_point_color, (int(round(projection(waypoints[point_number]*100,angles,origin,scale)[0])),int(round(projection(waypoints[point_number]*100,angles,origin,scale)[1]))), int(point_size*scale))
            elif point_number == len(waypoints)-1:
                pg.draw.circle(scr, end_point_color, (int(round(projection(waypoints[point_number]*100,angles,origin,scale)[0])),int(round(projection(waypoints[point_number]*100,angles,origin,scale)[1]))), int(point_size*scale))
            else:
                pg.draw.circle(scr, waypoint_color, (int(round(projection(waypoints[point_number]*100,angles,origin,scale)[0])),int(round(projection(waypoints[point_number]*100,angles,origin,scale)[1]))), int(point_size*scale))
            
            text = text_font.render(str(point_number), True, text_color)
            scr.blit(text,text.get_rect(center = projection(waypoints[point_number]*100,angles,origin,scale)))
            
            
    # draw path that drone travels
    if draw_path == True:
        path_points = np.vstack((path_points,drone_position))
        for point in path_points:
            pg.draw.circle(scr, path_color, (int(round(projection(point*100,angles,origin,scale)[0])),int(round(projection(point*100,angles,origin,scale)[1]))), int(path_point_size*scale))    


    #------------------------------ plot drone --------------------------------   
    # defines current position of rotors
    rotor_pos_wrt_drone = [np.dot(drone_rotation_matrix,[drone_size[0]/2,drone_size[1]/2,0]),np.dot(drone_rotation_matrix,[-drone_size[0]/2,-drone_size[1]/2,0]),np.dot(drone_rotation_matrix,[-drone_size[0]/2,drone_size[1]/2,0]),np.dot(drone_rotation_matrix,[drone_size[0]/2,-drone_size[1]/2,0])]
    rotor_pos = [drone_position*100+rotor_pos_wrt_drone[0],drone_position*100+rotor_pos_wrt_drone[1],drone_position*100+rotor_pos_wrt_drone[2],drone_position*100+rotor_pos_wrt_drone[3]]

    # draw drone facing direction
    pg.draw.line(scr, drone_facing_direction_color, projection(drone_position*100,angles,origin,scale), projection(drone_position*100+np.dot(drone_rotation_matrix,[drone_axis_length,0,0]),angles,origin,scale), 8)
  
    # draw drone diagonals
    pg.draw.line(scr, drone_diagonal_color, projection(rotor_pos[0],angles,origin,scale), projection(rotor_pos[1],angles,origin,scale), 8)
    pg.draw.line(scr, drone_diagonal_color, projection(rotor_pos[2],angles,origin,scale), projection(rotor_pos[3],angles,origin,scale), 8)

    # draw drone rotors
    for pos in rotor_pos:
        for theta in range(40):
            circle_pos = pos+np.dot(drone_rotation_matrix,np.array([sin(theta)*rotor_radius,cos(theta)*rotor_radius,0]))
            projected_circle_pos = projection(circle_pos,angles,origin,scale)
            projected_circle_pos = [int(round(projected_circle_pos[0])),int(round(projected_circle_pos[1]))]
            pg.draw.circle(scr, drone_rotor_color, projected_circle_pos, 0)
    
    #-------------------------- plot gates ------------------------------------
    if draw_gates == True:
        # visual imputs
        line_width = 4                                                                                            # in pixels
        gate_color = color.orange                                                                         # choose preferred color
        background_color = color.black                                                               # choose preferred color

        pixels_per_cm = 141.21*0.393701                                                          # = (pixels per inch) * (inch per cm)   , this is necessary for the field of view determination
        pygame_clamed_dimensions_per_actual_dimension = 1535/1920   # MUST be =1 for normal computer, but since pygame is not normal on my pc I have to assign this value

        # outputs
        actual_scale = scale/pixels_per_cm/pygame_clamed_dimensions_per_actual_dimension # something with no depth (x=0) will be scaled by this amount w.r.t. actual dimensions in the actual real world

        represented_size_in_screen = screensize/100/scale                              # The length of an object with the same size as the screen
        field_of_view = [2*np.arctan(represented_size_in_screen[0]/2),2*np.arctan(represented_size_in_screen[1]/2)]    # in rad
        
        # gate edge points                                                                                        # in cm w.r.t. coordinate system (which is centered in the middle of the screen)
        gate_points_1 = np.array([[0,50,50],[0,-50,50],[0,-50,-50],[0,50,-50]]) # defines 4 corners of a gate (in order of connection)

        # draw lines between points
        for gate_edge in [[0,1],[1,2],[2,3],[3,0]]:                                                      # draw gate defined by gates_points_1
            pg.draw.line(scr, gate_color, projection(gate_points_1[gate_edge[0]],angles,origin,scale), projection(gate_points_1[gate_edge[1]],angles,origin,scale), line_width)


    # draw axis of global coordinate system
    pg.draw.line(scr, axis_color, projection([0,0,0],angles,origin,scale), projection([axis_length,0,0],angles,origin,scale), 4)
    pg.draw.line(scr, axis_color, projection([0,0,0],angles,origin,scale), projection([0,axis_length,0],angles,origin,scale), 4)
    pg.draw.line(scr, axis_color, projection([0,0,0],angles,origin,scale), projection([0,0,axis_length],angles,origin,scale), 4)
    x_axis_text, y_axis_text, z_axis_text = text_font.render('X', True, text_color), text_font.render('Y', True, text_color), text_font.render('Z', True, text_color)
    scr.blit(x_axis_text,x_axis_text.get_rect(center = projection([axis_length+10,0,0],angles,origin,scale)))
    scr.blit(y_axis_text,y_axis_text.get_rect(center = projection([0,axis_length+10,0],angles,origin,scale)))
    scr.blit(z_axis_text,z_axis_text.get_rect(center = projection([0,0,axis_length+10],angles,origin,scale)))
    
    # update display
    pg.display.flip()
    # exit pygame
    for event in events:
        if event.type == pg.QUIT:
            pg.quit()
            pg.font.quit()
            running = False




