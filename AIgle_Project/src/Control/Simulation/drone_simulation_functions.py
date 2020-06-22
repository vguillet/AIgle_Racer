import pygame as pg
import numpy as np
from math import cos,sin

#------------------------------------ INITIAL VARIABLES -----------------------------------------
viewdistance = 10        # in m, MUST BE 1 for machine vision, should be ~10 for better view (also change scale to 10)

#--------------------------------------- FUNCTIONS ---------------------------------------------
"""
def rotation_matrix(angles, drone=False):
    x_angle,y_angle,z_angle = angles[0],angles[1],angles[2]
    x_rotation = np.array([[1,0,0],[0,cos(x_angle),-sin(x_angle)],[0,sin(x_angle),cos(x_angle)]])
    y_rotation = np.array([[cos(y_angle),0,sin(y_angle)],[0,1,0],[-sin(y_angle),0,cos(y_angle)]])
    z_rotation = np.array([[cos(z_angle),-sin(z_angle),0],[sin(z_angle),cos(z_angle),0],[0,0,1]])
    if drone == False:
        rot_mat = np.dot(np.dot(x_rotation,y_rotation),z_rotation)
    if drone == True:
        rot_mat = np.dot(np.dot(z_rotation,x_rotation),y_rotation)
    return rot_mat
"""
def rotation_matrix(angles, drone=False):
    si = angles[2]
    sigma = angles[0]
    theta = angles[1]
    return np.array([[cos(si)*cos(theta)-sin(sigma)*sin(si)*sin(theta), -cos(sigma)*sin(si), cos(si)*sin(theta)+cos(theta)*sin(sigma)*sin(si)],[cos(theta)*sin(si)+cos(si)*sin(sigma)*sin(theta),cos(sigma)*cos(si), sin(si)*sin(theta)-cos(si)*cos(theta)*sin(sigma)],[-cos(sigma)*sin(theta),sin(sigma),cos(sigma)*cos(theta)]])


def projection(point,angles,origin,scale):
    rotated_matrix = rotation_matrix(angles)
    rotated_point = np.dot(rotated_matrix,point)
    
    depth_scaling = 1/(viewdistance+rotated_point[0]/100)
    projection_matrix = np.array([[0,1*depth_scaling,0],[0,0,1*depth_scaling]])
    return scale*np.dot(projection_matrix,rotated_point)+origin


def normalize_angle(angle):   # from 0 to 2pi
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle


def pressed_keys():
    pressed_key = pg.key.get_pressed()
    keys = []
    if pressed_key[pg.K_UP]:
      keys.append('up')
    if pressed_key[pg.K_RIGHT]:
      keys.append('right')
    if pressed_key[pg.K_DOWN]:
      keys.append('down')
    if pressed_key[pg.K_LEFT]:
      keys.append('left')
    if pressed_key[pg.K_w]:
      keys.append('w')
    if pressed_key[pg.K_d]:
      keys.append('d')
    if pressed_key[pg.K_s]:
      keys.append('s')
    if pressed_key[pg.K_a]:
      keys.append('a')
    if pressed_key[pg.K_q]:
      keys.append('q')
    if pressed_key[pg.K_e]:
      keys.append('e')
    return keys

class colors:
  def __init__(self,white,lightgrey,grey,black,brown,red,orange,yellow,lime,green,cyan,lightblue,blue,purple,magenta,pink):
    self.white = white
    self.lightgrey = lightgrey
    self.grey = grey
    self.black = black
    self.brown = brown
    self.red = red
    self.orange = orange
    self.yellow = yellow
    self.lime = lime
    self.green = green
    self.cyan = cyan
    self.lightblue = lightblue
    self.blue = blue
    self.purple = purple
    self.magenta = magenta
    self.pink = pink
color = colors((255,255,255),(208,208,208),(138,138,138),(59,59,64),(156,113,83),(181,64,64),(234,173,83),(234,234,77),(149,218,65),(100,129,56),(57,123,149),(150,185,234),(60,109,181),(189,124,221),(225,143,218),(240,181,211))
