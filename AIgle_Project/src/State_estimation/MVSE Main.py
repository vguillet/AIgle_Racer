import numpy as np
from matplotlib.image import imread
import random as rnd
import cProfile
from math import *
import scipy.optimize
import matplotlib.pyplot as plt
#STANDARD GLOBAL VARIABLES
#Axis system: the x-axis points through the gate, the y-axis points "up", the z-axis points to the side, parallel to the gate
imageCurrent = imread(r"C:\Users\Max van Huffelen\Desktop\Generic Junk\DSE\MVSE\Verification_Image_Generated_with_noise.png")
color_max = [1.1*234/255,1.1*173/255,1.1*83/255] #the upper bound for the color of the (detected) gate
color_min = [0.9*234/255,0.9*173/255,0.9*83/255]   #the lower bound of the color of the (detected) gate
gate_width = 1 #gate width in meters
gate_height = 1 #gate height in meters
camera_angle_horizontal = 2.651635327336065 #horizontal camera angle in radians
camera_angle_vertical = 2.651635327336065 #vertical camera angle in radians
time_between_images = 1/60 #time between instances of images in seconds


# location_history = open("location_history.txt", "r+")
# #location_history.write('0,0,4\n')     #add an initial value for the location wrt the gate. This should be changed later to the actual initial distance to the gate
# print(location_history.readlines()[-1].strip().split(','))

class pixel:
	def __init__(self, x_pix, y_pix, image = imageCurrent, dir = None):
		self.x_pix = x_pix
		self.y_pix = y_pix
		self.image = image

		if dir == None:
			pass
		elif dir == 'UP':
			self.SnakeUp()
		elif dir == 'DOWN':
			self.SnakeDown()
		elif dir == 'LEFT':
			self.SnakeLeft()
		elif dir == 'RIGHT':
			self.SnakeRight()

	def CheckSelf(self):
		#check if current pixel is of the desired color
		#if self.image[self.y_pix, self.x_pix] is color:
		if color_max[0] >= self.image[self.y_pix, self.x_pix, 0] >= color_min[0] and color_max[1] >= self.image[self.y_pix, self.x_pix, 1] >= color_min[1] and color_max[2] >= self.image[self.y_pix, self.x_pix, 2] >= color_min[2]:
			return True
		else:
			return False

	def CheckPixel(self, dy, dx):
		try:
			if color_max[0] >= self.image[self.y_pix+dy, self.x_pix+dx, 0] >= color_min[0] and color_max[1] >= self.image[self.y_pix+dy, self.x_pix+dx, 1] >= color_min[1] and color_max[2] >= self.image[self.y_pix+dy, self.x_pix+dx, 2] >= color_min[2]:
				if self.y_pix+dy >= 0 and self.x_pix+dx >= 0:
					return True
			else:
				return False
		except:
			return False

	def CheckPixelAbove(self):
		# Checks the pixel directly above, to the above left and above right for if they are shaded red (normally you'd use " if self.image[self.y_pix -1, self.x_pix] is color " or something similar)
		# then, returns a string depending on where a pixel of the specified color is found. If none are found, false is returned instead.
		if self.CheckPixel(-1, 0):
			return 'ABOVE'
		elif self.CheckPixel(-2, -1):
			return 'ABOVE LEFT'
		elif self.CheckPixel(-2, 1):
			return 'ABOVE RIGHT'
		else:
			return False

	def SnakeUp(self):
		#Checks the above pixels, then if any pixels are found, the pixel object's location is moved up, and the program is repeated.
		#if no new pixels are found, the snaking is finished and the current location should be the highest point
		nextPixel = 'not false'     #I recognise that I am not very funny. I just need a variable to definte nextPixel and an arbitrary string is cheaper than running e.g. self.CheckPixelAbove()
		while nextPixel is not False:
			nextPixel = self.CheckPixelAbove()
			if nextPixel == 'ABOVE':
				self.y_pix += -1
			elif nextPixel == 'ABOVE LEFT':
				self.y_pix += -2
				self.x_pix += -1
			elif nextPixel == 'ABOVE RIGHT':
				self.y_pix += -2
				self.x_pix += 1
			elif nextPixel is False:
				# print('Finished snaking up at', self.y_pix, self.x_pix)
				return True
			else:
				#An error must've been encountered - this isn't supposed to happen
				return False

	def CheckPixelBelow(self):
		if self.CheckPixel(1, 0):
			return 'BELOW'
		elif self.CheckPixel(2, -1):
			return 'BELOW LEFT'
		elif self.CheckPixel(2, 1):
			return 'BELOW RIGHT'
		else:
			return False

	def SnakeDown(self):
		nextPixel = 'not false'     #I recognise that I am not very funny - I just need some variable for nextPixel to initiate the looop
		while nextPixel is not False:
			nextPixel = self.CheckPixelBelow()
			if nextPixel == 'BELOW':
				self.y_pix += 1
			elif nextPixel == 'BELOW LEFT':
				self.y_pix += 2
				self.x_pix += -1
			elif nextPixel == 'BELOW RIGHT':
				self.y_pix += 2
				self.x_pix += 1
			elif nextPixel is False:
				# print('Finished snaking down at', self.y_pix, self.x_pix)
				return True
			else:
				# print("An error must've been encountered - this isn't supposed to happen")
				return False

	def CheckPixelLeft(self):
		if self.CheckPixel(0, -1):
			return 'LEFT'
		elif self.CheckPixel(-1, -2):
			return 'ABOVE LEFT'
		elif self.CheckPixel(1, -2):
			return 'BELOW LEFT'
		else:
			return False

	def SnakeLeft(self):
		nextPixel = 'not false'     #I recognise that I am not very clever/funny, I just need a value for nextPixel to initialize the loop
		while nextPixel is not False:
			nextPixel = self.CheckPixelLeft()
			if nextPixel == 'LEFT':
				self.x_pix += -1
			elif nextPixel == 'ABOVE LEFT':
				self.y_pix += -1
				self.x_pix += -2
			elif nextPixel == 'BELOW LEFT':
				self.y_pix += +1
				self.x_pix += -2
			elif nextPixel is False:
				# print('Finished snaking left at', self.y_pix, self.x_pix)
				return True
			else:
				# print("An error must've been encountered - this isn't supposed to happen")
				return False

	def CheckPixelRight(self):
		if self.CheckPixel(0, 1):
			return 'RIGHT'
		elif self.CheckPixel(-1, 2):
			return 'ABOVE RIGHT'
		elif self.CheckPixel(1, 2):
			return 'BELOW RIGHT'
		else:
			return False

	def SnakeRight(self):
		nextPixel = 'not false'     #I recognise that I am not very funny, I just need some dummy value to initialise the loop
		while nextPixel is not False:
			nextPixel = self.CheckPixelRight()
			if nextPixel == 'RIGHT':
				self.x_pix += 1
			elif nextPixel == 'ABOVE RIGHT':
				self.y_pix += -1
				self.x_pix += 2
			elif nextPixel == 'BELOW RIGHT':
				self.y_pix += 1
				self.x_pix += 2
			elif nextPixel is False:
				# print('Finished snaking right at', self.y_pix, self.x_pix)
				return True
			else:
				# print("An error must've been encountered - this isn't supposed to happen")
				return False


def recognizeImage(desired_color, image = imageCurrent):

	#keep loop running until the gate is found
	foundGate = False
	while foundGate is False:
		# generate 'random' positions
		position_x = rnd.randint(0, imageCurrent.shape[1] - 1) #recommended point to start: 325
		position_y = rnd.randint(0, imageCurrent.shape[0] - 1) #recommended point to start: 500
		# print(position_y, position_x)
		samplePoint = pixel(position_x, position_y)
		if samplePoint.CheckSelf() is True:         #Check if the point is of the correct color
			#print(samplePoint.image[samplePoint.y_pix, samplePoint.x_pix])
			upPoint = pixel(position_x, position_y, dir = 'UP')
			downPoint = pixel(position_x, position_y, dir = 'DOWN')
			#if (downPoint.y_pix - upPoint.y_pix) >= 10: #Check if the distance between the upper and lower part is at least X pixels
			up_leftPoint = pixel(upPoint.x_pix, upPoint.y_pix, dir = 'LEFT')
			up_rightPoint = pixel(upPoint.x_pix, upPoint.y_pix, dir = 'RIGHT')
			down_leftPoint = pixel(downPoint.x_pix, downPoint.y_pix, dir = 'LEFT')
			down_rightPoint = pixel(downPoint.x_pix, downPoint.y_pix, dir = 'RIGHT')
			if (up_rightPoint.x_pix - down_leftPoint.x_pix) > 10: #check if the width is at least X pixels
				up_leftPoint = pixel(up_leftPoint.x_pix, up_leftPoint.y_pix, dir = 'UP')
				up_rightPoint = pixel(up_rightPoint.x_pix, up_rightPoint.y_pix, dir = 'UP')
				down_leftPoint = pixel(down_leftPoint.x_pix, down_leftPoint.y_pix, dir = 'DOWN')
				down_rightPoint = pixel(down_rightPoint.x_pix, down_rightPoint.y_pix, dir = 'DOWN')
				if (down_rightPoint.y_pix - up_rightPoint.y_pix) >= 10 and (down_leftPoint.y_pix - up_leftPoint.y_pix) >= 10: #check if the vertical distance between the corners is at least X pixels
					gate_points = dict()
					gate_points['up_left'] = up_leftPoint #[up_leftPoint.x_pix, up_leftPoint.y_pix]
					gate_points['up_right'] = up_rightPoint #[up_rightPoint.x_pix, up_rightPoint.y_pix]
					gate_points['down_left'] = down_leftPoint #[down_leftPoint.x_pix, down_leftPoint.y_pix]
					gate_points['down_right'] = down_rightPoint #[down_rightPoint.x_pix, down_rightPoint.y_pix]
					foundGate = True



					return gate_points

def pixel2ray(point, image = imageCurrent):
	#returns the slope of the ray through a pixel. remember y is positive downward!
	pixel_x, pixel_y = point.x_pix, point.y_pix
	N_y, N_x = image.shape[0], image.shape[1]
	slope_x = (2*pixel_x/N_x - 1)*tan(camera_angle_horizontal/2)
	slope_y = (1 - 2*pixel_y/N_y)*tan(camera_angle_vertical/2)
	# slope_x = tan((pixel_x/N_x - 0.5)*camera_angle_horizontal)
	# slope_y = -tan((pixel_y/N_x - 0.5)*camera_angle_vertical)
	return slope_y, slope_x

def distance_squared(x1, y1, z1, x2, y2, z2):
	return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

# def findGateDistanceOld(up_leftPoint, up_rightPoint, down_leftPoint, down_rightPoint, image = imageCurrent):
# 	up_leftSlope_x,up_leftSlope_y = pixel2ray(up_leftPoint)
# 	up_rightSlope_x,up_rightSlope_y = pixel2ray(up_rightPoint)
# 	down_leftSlope_x,down_leftSlope_y = pixel2ray(down_leftPoint)
# 	down_rightSlope_x,down_rightSlope_y = pixel2ray(down_rightPoint)
# 	def solve_fixedDistances(points):
# 		up_left_z, up_right_z, down_left_z, down_right_z = points
# 		up_horizontal = distance_squared(up_leftSlope_x*up_left_z, up_leftSlope_y*up_left_z, up_left_z, up_rightSlope_x*up_right_z, up_rightSlope_y*up_right_z, up_right_z) - gate_width**2
# 		down_horizontal = distance_squared(down_leftSlope_x*down_left_z, down_leftSlope_y*down_left_z, down_left_z, down_rightSlope_x*down_right_z, down_rightSlope_y*down_right_z, down_right_z) - gate_width**2
# 		left_vertical = distance_squared(up_leftSlope_x*up_left_z, up_leftSlope_y*up_left_z, up_left_z, down_leftSlope_x*down_left_z, down_leftSlope_y*down_left_z, down_left_z) - gate_height**2
# 		right_vertical = distance_squared(up_rightSlope_x*up_right_z, up_rightSlope_y*up_right_z, up_right_z, down_rightSlope_x*down_right_z, down_rightSlope_y*down_right_z, down_right_z) - gate_height**2
# 		up_right_diagonal = distance_squared(down_leftSlope_x*down_left_z, down_leftSlope_y*down_left_z, down_left_z, up_rightSlope_x*up_right_z, up_rightSlope_y*up_right_z, up_right_z) - (gate_height**2 + gate_width**2)
# 		up_left_diagonal = distance_squared(down_rightSlope_x*down_right_z, down_rightSlope_y*down_right_z, down_right_z, up_leftSlope_x*up_left_z, up_leftSlope_y*up_left_z, up_left_z) - (gate_width**2 + gate_height**2)
# 		return up_horizontal, down_horizontal, left_vertical, right_vertical, up_right_diagonal, up_left_diagonal
#
# 	# up_left_z, up_right_z, down_left_z, down_right_z = scipy.optimize.fsolve(solve_fixedDistances, [2, 2, 2, 2])
# 	up_left_z, up_right_z, down_left_z, down_right_z = scipy.optimize.root(solve_fixedDistances, [2, 2, 2, 2], method = 'lm')['x']
# 	up_left_point = (up_leftSlope_x*up_left_z, up_leftSlope_y*up_left_z, up_left_z)
# 	up_right_point = (up_rightSlope_x*up_right_z, up_rightSlope_y*up_right_z, up_right_z)
# 	down_left_point = (down_leftSlope_x*down_left_z, down_leftSlope_y*down_left_z, down_left_z)
# 	down_right_point = (down_rightSlope_x*down_right_z, down_rightSlope_y*down_right_z, down_right_z)
# 	return up_left_point, up_right_point, down_left_point, down_right_point

# def findAngle(p_main, p1, p2):
# 	#Takes one center point (pixel class) and two others, then calculates the angle p1-p_main-p2 (so draw lines from p1 and p2 to p_main, then find the angle between the lines)
# 	x_1 = p1.x_pix - p_main.x_pix
# 	y_1 = p1.y_pix - p_main.y_pix
# 	x_2 = p2.x_pix - p_main.x_pix
# 	y_2 = p2.y_pix - p_main.y_pix
# 	return acos((x_1*x_2 + y_1*y_2)/(sqrt(x_1**2+y_1**2)*sqrt(x_2**2+y_2**2)))
#
# def findGateOrientation(up_left_pixel, up_right_pixel, down_right_pixel, down_left_pixel):
# 	#find angles in each of the four corners
# 	up_left_angle = findAngle(up_left_pixel, up_right_pixel, down_left_pixel)
# 	up_right_angle = findAngle(up_right_pixel, up_left_pixel, down_right_pixel)
# 	down_right_angle = findAngle(down_right_pixel, up_right_pixel, down_left_pixel)
# 	down_left_angle = findAngle(down_left_pixel, up_left_pixel, down_right_pixel)
# 	#use the angles to find the angles by which each side widens
# 	alpha = 0.5*up_left_angle + 0.5*up_right_angle - pi/2 #vertical angle (angle by which the square becomes wider towards the bottom of the square)
# 	beta = pi/2 - up_right_angle + alpha #horizontal angle; the angle by which the square becomes taller further to the right
#
# 	#find the angle by which the square is rotated (w.r.t horizontal, counter-clockwise positive )
# 	rotation_angle = atan((up_right_pixel.y_pix - up_left_pixel.y_pix)/(up_right_pixel.x_pix-up_left_pixel.x_pix)) - beta #the angle by which the square is rotated; ccw positive
#
# 	#'De-rotate' the gate
# 	def rotatePointClockwise(point, angle):
# 		new_x = point.x_pix*cos(angle) + point.y_pix*sin(angle)
# 		new_y = point.y_pix*cos(angle) - point.x_pix*sin(angle)
# 		return new_y, new_x
#
# 	up_left_y, up_left_x = rotatePointClockwise(up_left_pixel, rotation_angle)
# 	up_right_y, up_right_x = rotatePointClockwise(up_right_pixel, rotation_angle)
# 	down_right_y, down_right_x = rotatePointClockwise(down_right_pixel, rotation_angle)
# 	down_left_y, down_left_x = rotatePointClockwise(down_left_pixel, rotation_angle)
#
# 	#Find the scaeling factors
# 	scale_factor_down = (down_right_x - down_left_x)/(up_right_x - up_left_x)
# 	scale_factor_right = (up_right_y - down_right_y)/(up_left_y - down_left_y)
#
# 	return scale_factor_down, scale_factor_right, rotation_angle
#
# def findGateDistance(up_left_slope_y, up_left_slope_x, down_right_slope_y, down_right_slope_x, scale_factor_down, scale_factor_right):
# 	def distanceUpHorizontal(up_left_z):
# 		up_left_x = up_left_slope_x*up_left_z
# 		up_left_y = up_left_slope_y*up_left_z
# 		down_right_z = up_left_z/(scale_factor_right*scale_factor_down)
# 		down_right_x = down_right_slope_x*down_right_z
# 		down_right_y = down_right_slope_y*down_right_z
# 		return distance_squared(up_left_x, up_left_y, up_left_z, down_right_x, down_right_y, down_right_z) - gate_width**2 - gate_height**2
# 	up_left_z = float(scipy.optimize.fsolve(distanceUpHorizontal, 2))
# 	up_right_z = up_left_z / scale_factor_right
# 	down_left_z = up_left_z / scale_factor_down
# 	down_right_z = up_left_z / (scale_factor_right * scale_factor_down)
# 	return up_left_z, up_right_z, down_right_z, down_left_z
#
# def findGateCoordinatesRotation(color_min = "foo color", color_max = "foo color", image = imageCurrent):
# 	#1 Find the four corner points
# 	gate_points = recognizeImage("foo color")
#
# 	#2a Find the rays going through the points
# 	up_left_slope_y, up_left_slope_x = pixel2ray(gate_points["up_left"])
# 	up_right_slope_y, up_right_slope_x = pixel2ray(gate_points["up_right"])
# 	down_right_slope_y, down_right_slope_x = pixel2ray(gate_points["down_right"])
# 	down_left_slope_y, down_left_slope_x = pixel2ray(gate_points["down_left"])
#
# 	#2b Find the rotation of the gate
# 	scale_factor_down, scale_factor_right, rotation_angle = findGateOrientation(gate_points["up_left"], gate_points["up_right"], gate_points["down_right"], gate_points["down_left"])
#
# 	#3 Find the locations of the corners and central point
# 	up_left_z, up_right_z, down_right_z, down_left_z = findGateDistance(up_left_slope_y, up_left_slope_x, up_right_slope_y, up_right_slope_x, scale_factor_down, scale_factor_right)
# 	up_left_x, up_left_y = up_left_z*up_left_slope_x, up_left_z*up_left_slope_y
# 	up_right_x, up_right_y = up_right_z*up_right_slope_x, up_right_z*up_right_slope_y
# 	down_right_x, down_right_y = down_right_z*down_right_slope_x, down_right_z*down_right_slope_y
# 	down_left_x, down_left_y = down_left_z*down_left_slope_x, down_left_z*down_left_slope_y
# 	central_x, central_y, central_z = (up_left_x + up_right_x + down_right_x + down_left_x)/4, (up_left_y + up_right_y + down_right_y + down_left_y)/4, (up_left_z + up_right_z + down_right_z + down_left_z)/4
#
# 	#4a Find the current velocity
# 	location_history = open("location_history.txt", "r+")
# 	old_values = location_history.readlines()[-1].strip().split(',')
# 	old_x, old_y, old_z = float(old_values[0]), float(old_values[1]), float(old_values[2])
# 	velocity_x, velocity_y, velocity_z = (central_x - old_x)/time_between_images, (central_y - old_y)/time_between_images, (central_z - old_z)/time_between_images
#
# 	#4b Append the found location to the location_history file so it can be used later to find future velocities
# 	location_history.write(f'\n{central_x},{central_y},{central_z}')
#
# 	#5 find the orthonormal vector to the gate plane (estimate)
# 	x_1 = down_left_x - central_x
# 	y_1 = down_left_y - central_y
# 	z_1 = down_left_z - central_z
# 	x_2 = down_right_x - central_x
# 	y_2 = down_right_y - central_y
# 	z_2 = down_right_z - central_z
#
# 	orthogonal_vector = np.array([y_1*z_2 - z_1*y_2, x_1*z_2 - z_1*x_2, x_1*y_2 - x_2*y_1])
# 	orthogonal_vector /= np.linalg.norm(orthogonal_vector)
#
#
# 	#(Optional) Print variables for verification and debugging
# 	print(f'Up-left; x: {up_left_x}, y: {up_left_y}, z: {up_left_z}')
# 	print(f'Up-right; x: {up_right_x}, y: {up_right_y}, z: {up_right_z}')
# 	print(f'Down-right; x: {down_right_x}, y: {down_right_y}, z: {down_right_z}')
# 	print(f'Down-left; x: {down_left_x}, y: {down_left_y}, z: {down_left_z}')
# 	print(f'Scale factor down: {scale_factor_down}')
# 	print(f'Scale factor right: {scale_factor_right}')
# 	print(f"Pixel Up-Left at {gate_points['up_left'].x_pix}, {gate_points['up_left'].y_pix}")
# 	print(f"Pixel Up-Right at {gate_points['up_right'].x_pix}, {gate_points['up_right'].y_pix}")
# 	print(f"Pixel Down-Right at {gate_points['down_right'].x_pix}, {gate_points['down_right'].y_pix}")
# 	print(f"Pixel Down-Left at {gate_points['down_left'].x_pix}, {gate_points['down_left'].y_pix}")
# 	print(f"Up-left slope; y: {up_left_slope_y}, x: {up_left_slope_x}")
# 	#6 plot the gate
# 	fig = plt.figure()
# 	ax = plt.axes(projection='3d')
# 	ax.set_xlabel('x')
# 	ax.set_ylabel('y')
# 	ax.set_zlabel('z')
# 	ax.plot3D([up_left_x, up_right_x, down_right_x, down_left_x, up_left_x], [up_left_y, up_right_y, down_right_y, down_left_y, up_left_y], [up_left_z, up_right_z, down_right_z, down_left_z, up_left_z], 'gray')
# 	plt.show()
# 	return (central_x, central_y, central_z), (velocity_x, velocity_y, velocity_z), orthogonal_vector

#print(findGateCoordinatesRotation())

def dist_to_point(x, y, z):
	return sqrt(x**2+y**2+z**2)

def vectors2angle(x1, y1, z1, x2, y2, z2):
	return acos((x1*x2 + y1*y2 + z1*z2)/(sqrt(x1**2 + y1**2 + z1**2) * sqrt(x2**2 + y2**2 + z2**2)))

def p4p():

	#1 Find the four corner points
	gate_points = recognizeImage("foo color")

	#2a Find the rays going through the points
	up_left_slope_y, up_left_slope_x = pixel2ray(gate_points["up_left"])
	up_right_slope_y, up_right_slope_x = pixel2ray(gate_points["up_right"])
	down_right_slope_y, down_right_slope_x = pixel2ray(gate_points["down_right"])
	down_left_slope_y, down_left_slope_x = pixel2ray(gate_points["down_left"])

	#2b Find the angles between the rays
		#Since the length of the vector doesn't matter for the angle between them, we make a vector with z = 1, meaning that the x and y components are the slope times 1
	up_angle = vectors2angle(up_left_slope_x, up_left_slope_y, 1, up_right_slope_x, up_right_slope_y, 1)
	right_angle = vectors2angle(up_right_slope_x, up_right_slope_y, 1, down_right_slope_x, down_right_slope_y, 1)
	down_angle = vectors2angle(down_right_slope_x, down_right_slope_y, 1, down_left_slope_x, down_left_slope_y, 1)
	left_angle = vectors2angle(down_left_slope_x, down_left_slope_y, 1, up_left_slope_x, up_left_slope_y, 1)
	diagonal_up_right_angle = vectors2angle(down_left_slope_x, down_left_slope_y, 1, up_right_slope_x, up_right_slope_y, 1)
	diagonal_up_left_angle = vectors2angle(down_right_slope_x, down_right_slope_y, 1, up_left_slope_x, up_left_slope_y, 1)

	#3 Find the distance to each point using PnP
	def PnP(distances):
		up_left_dist, up_right_dist, down_right_dist, down_left_dist = distances
		x1 = up_left_dist**2 + up_right_dist**2 - 2*cos(up_angle)*up_left_dist*up_right_dist - gate_width**2
		x2 = up_right_dist**2 + down_right_dist**2 - 2*cos(right_angle)*up_right_dist*down_right_dist - gate_height**2
		x3 = down_right_dist**2 + down_left_dist**2 - 2*cos(down_angle)*down_right_dist*down_left_dist - gate_width**2
		x4 = down_left_dist**2 + up_left_dist**2 - 2*cos(left_angle)*down_left_dist*up_left_dist - gate_height**2
		x5 = down_left_dist**2 + up_right_dist**2 - 2*cos(diagonal_up_right_angle)*down_left_dist*up_right_dist - gate_height**2 - gate_width**2
		x6 = down_right_dist**2 + up_left_dist**2 - 2*cos(diagonal_up_left_angle)*down_right_dist*up_left_dist - gate_height**2 - gate_width**2
		return x1,x2,x3,x4,x5,x6

	distances = scipy.optimize.root(PnP, [2,2,2,2], method='lm')['x']

	up_left_dist, up_right_dist, down_right_dist, down_left_dist = distances
	print('Remainder in PnP function; ', PnP(distances))
	print(f'Distances found; {up_left_dist}, {up_right_dist}, {down_right_dist}, {down_left_dist}')

	#4 Find the actual locations of the corners
	def decomposeVector(vector_length, slope_x, slope_y):
		x = sin(atan(slope_x)) * cos(atan(slope_y)) * vector_length
		y = sin(atan(slope_y)) * cos(atan(slope_x)) * vector_length
		z = cos(atan(slope_x)) * cos(atan(slope_y)) * vector_length
		return x, y, z

	up_left_x, up_left_y, up_left_z = decomposeVector(up_left_dist, up_left_slope_x, up_left_slope_y)
	up_right_x, up_right_y, up_right_z = decomposeVector(up_right_dist, up_right_slope_x, up_right_slope_y)
	down_right_x, down_right_y, down_right_z = decomposeVector(down_right_dist, down_right_slope_x, down_right_slope_y)
	down_left_x, down_left_y, down_left_z = decomposeVector(down_left_dist, down_left_slope_x, down_left_slope_y)
	center_x, center_y, center_z = (up_left_x + up_right_x + down_right_x + down_left_x)/4, (up_left_y + up_right_y + down_right_y + down_left_y)/4, (up_left_z + up_right_z + down_right_z + down_left_z)/4

	#5 find the orthonormal vector to the gate plane (estimate)
	x_1 = down_left_x - center_x
	y_1 = down_left_y - center_y
	z_1 = down_left_z - center_z
	x_2 = down_right_x - center_x
	y_2 = down_right_y - center_y
	z_2 = down_right_z - center_z

	orthogonal_vector = np.array([y_1*z_2 - z_1*y_2, x_1*z_2 - z_1*x_2, x_1*y_2 - x_2*y_1])
	orthogonal_vector /= np.linalg.norm(orthogonal_vector)

	#6a Find the current velocity
	location_history = open("location_history.txt", "r+")
	old_values = location_history.readlines()[-1].strip().split(',')
	old_x, old_y, old_z = float(old_values[0]), float(old_values[1]), float(old_values[2])
	velocity_x, velocity_y, velocity_z = (center_x - old_x)/time_between_images, (center_y - old_y)/time_between_images, (center_z - old_z)/time_between_images

	#6b Append the found location to the location_history file so it can be used later to find future velocities
	location_history.write(f'\n{center_x},{center_y},{center_z}')

	#(Optional) Print variables for verification and debugging
	if True:
		print(f'Up-left; x: {up_left_x}, y: {up_left_y}, z: {up_left_z}')
		print(f'Up-right; x: {up_right_x}, y: {up_right_y}, z: {up_right_z}')
		print(f'Down-right; x: {down_right_x}, y: {down_right_y}, z: {down_right_z}')
		print(f'Down-left; x: {down_left_x}, y: {down_left_y}, z: {down_left_z}')
		print(f'Center; x: {center_x}, y: {center_y}, z: {center_z}')
		print(f"Pixel Up-Left at {gate_points['up_left'].x_pix}, {gate_points['up_left'].y_pix}")
		print(f"Pixel Up-Right at {gate_points['up_right'].x_pix}, {gate_points['up_right'].y_pix}")
		print(f"Pixel Down-Right at {gate_points['down_right'].x_pix}, {gate_points['down_right'].y_pix}")
		print(f"Pixel Down-Left at {gate_points['down_left'].x_pix}, {gate_points['down_left'].y_pix}")
		print(f"Up-left slope; y: {up_left_slope_y}, x: {up_left_slope_x}")
		print(f'Up-right slope; y: {up_right_slope_y}, x: {up_right_slope_x}')
		print(f'Down-right slope; y: {down_right_slope_y}, x: {down_right_slope_x}')
		print(f'Down-left slope; y: {down_left_slope_y}, x: {down_left_slope_x}')

	#(Optional) plot the detected gate corners
	if True:
		implot = plt.imshow(imageCurrent)
		x_list = [gate_points['up_left'].x_pix, gate_points['up_right'].x_pix, gate_points['down_right'].x_pix, gate_points['down_left'].x_pix, gate_points['up_left'].x_pix]
		y_list = [gate_points['up_left'].y_pix,gate_points['up_right'].y_pix,gate_points['down_right'].y_pix,gate_points['down_left'].y_pix,gate_points['up_left'].y_pix]
		plt.scatter(x_list, y_list, marker = 'o', color = 'cyan')
		plt.axis([0, imageCurrent.shape[0]-1, 0, imageCurrent.shape[1]-1])
		implot.axes.xaxis.set_visible(False)
		implot.axes.yaxis.set_visible(False)
		plt.plot()
		plt.savefig(r"C:\Users\Max van Huffelen\Desktop\Generic Junk\DSE\MVSE\Verification_Image_Detected.png")
		plt.show()
	#(Optional) plot the gate
	if True:
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.plot3D([up_left_x, up_right_x, down_right_x, down_left_x, up_left_x], [up_left_y, up_right_y, down_right_y, down_left_y, up_left_y], [up_left_z, up_right_z, down_right_z, down_left_z, up_left_z], 'gray')
		ax.plot3D([0],[0],[0], 'gray')
		plt.show()
		plt.savefig(r"C:\Users\Max van Huffelen\Desktop\Generic Junk\DSE\MVSE\Gate_Location_Estimated.png" , origin= 'upper')
	return (center_x, down_left_y, down_left_z), (velocity_x, velocity_y, velocity_z), orthogonal_vector



#### TESTING BELOW HERE ####


#Testing PnP
ulx, uly, ulz = (-1, 1, 1)
urx, ury, urz = (1, 1, 1)
drx, dry, drz = (1, -1, 1)
dlx, dly, dlz = (-1, -1, 1)
au = vectors2angle(ulx, uly, ulz, urx, ury, urz)
ad = vectors2angle(drx, dry, drz, dlx, dly, dlz)
al = vectors2angle(dlx, dly, dlz, ulx, uly, ulz)
ar = vectors2angle(urx, ury, urz, drx, dry, drz)
adur = vectors2angle(dlx, dly, dlz, urx, ury, urz)
adul = vectors2angle(drx, dry, drz, ulx, uly, ulz)

up_left_slope_y, up_left_slope_x = uly/ulz, ulx/ulz
up_right_slope_y, up_right_slope_x = ury/urz, urx/urz
down_right_slope_y, down_right_slope_x = dry/drz, drx/drz
down_left_slope_y, down_left_slope_x = dly/dlz, dlx/dlz

def PnP(distances):
	up_left_dist, up_right_dist, down_right_dist, down_left_dist = distances
	x1 = up_left_dist ** 2 + up_right_dist ** 2 - 2 * cos(au) * up_left_dist * up_right_dist - gate_width ** 2
	x2 = up_right_dist ** 2 + down_right_dist ** 2 - 2 * cos(ar) * up_right_dist * down_right_dist - gate_height ** 2
	x3 = down_right_dist ** 2 + down_left_dist ** 2 - 2 * cos(ad) * down_right_dist * down_left_dist - gate_width ** 2
	x4 = down_left_dist ** 2 + up_left_dist ** 2 - 2 * cos(al) * down_left_dist * up_left_dist - gate_height ** 2
	x5 = down_left_dist ** 2 + up_right_dist ** 2 - 2 * cos(adur) * down_left_dist * up_right_dist - gate_height ** 2 - gate_width ** 2
	x6 = down_right_dist ** 2 + up_left_dist ** 2 - 2 * cos(adul) * down_right_dist * up_left_dist - gate_height ** 2 - gate_width ** 2
	return x1, x2, x3, x4, x5, x6
distances = scipy.optimize.root(PnP, [2, 2, 2, 2], method='lm')['x']

up_left_dist, up_right_dist, down_right_dist, down_left_dist = distances


def decomposeVector(vector_length, slope_x, slope_y):
	x = sin(atan(slope_x)) * cos(atan(slope_y)) * vector_length
	y = sin(atan(slope_y)) * cos(atan(slope_x)) * vector_length
	z = cos(atan(slope_x)) * cos(atan(slope_y)) * vector_length
	return x, y, z

up_left_x, up_left_y, up_left_z = decomposeVector(up_left_dist, up_left_slope_x, up_left_slope_y)
up_right_x, up_right_y, up_right_z = decomposeVector(up_right_dist, up_right_slope_x, up_right_slope_y)
down_right_x, down_right_y, down_right_z = decomposeVector(down_right_dist, down_right_slope_x, down_right_slope_y)
down_left_x, down_left_y, down_left_z = decomposeVector(down_left_dist, down_left_slope_x, down_left_slope_y)
center_x, center_y, center_z = (up_left_x + up_right_x + down_right_x + down_left_x) / 4, (up_left_y + up_right_y + down_right_y + down_left_y) / 4, (up_left_z + up_right_z + down_right_z + down_left_z) / 4


# print(f'distances found; {up_left_dist}, {up_right_dist}, {down_left_dist}, {down_right_dist}')
# print(f'real distances; {dist_to_point(ulx, uly, ulz)}, {dist_to_point(urx, ury, urz)}, {dist_to_point(dlx, dly, dlz)}, {dist_to_point(drx, dry, drz)}')
# print(f'locations; up-left: {up_left_x, up_left_y, up_left_z}\n up-right: {up_right_x, up_right_y, up_right_z}\n down-left: {down_right_x, down_right_y, down_right_z}\n down-right: {down_left_x, down_left_y, down_left_z}')

p4p()
# print(f'Actual distances: {dist_to_point(ulx, uly, ulz)}, {dist_to_point(urx, ury, urz)}, {dist_to_point(drx, dry, drz)}, {dist_to_point(dlx, dly, dlz)}')









