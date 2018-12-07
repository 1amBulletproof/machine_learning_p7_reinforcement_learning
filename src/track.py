#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			12/5/2018
#@description	Track encapsulation

import argparse
import numpy as np

#=============================
# Track
#
#	- Class to encapsulate track
#	- '#' = wall, '.' = track, 'S' = start, 'F' = finish
#=============================
class Track():

	#=============================
	# __init__()
	#	- Initialize the track data via file and/or copy
	#	- find relevant areas
	#@return	list of starting points as tuples
	#=============================
	def __init__(self, file_name, data=None):
		self.TRK_CHAR = '.'
		self.WALL_CHAR = '#'
		self.START_CHAR = 'S'
		self.END_CHAR = 'F'

		self.file_name = file_name
		if data == None:
			#Get data from file
			self.data = self.read_track_as_2darray(self.file_name)
			self.np_data = np.array(self.data, dtype='str')
		else:
			self.data = data

		self.shape = (len(self.data), len(self.data[0]))
		self.start_points = self.find_starting_points(self.data)
		self.finish_points = self.find_finish_points(self.data)
		self.finish_line = self.find_finish_line() #Find the x-range & y-range of the wall
		self.wall_points = self.find_wall_points(self.data)
		self.track_points = self.find_track_points(self.data)
		self.valid_points = self.track_points + self.start_points

	#=============================
	# print_track()
	#	- print the track
	#=============================
	def print_track(self):
		for lines in self.data:
			print(''.join(str(n) for n in lines))

	#=============================
	# read_track_as_1darray()
	#	- read the track from a given file
	#@param		file_name
	#@return	data as 1d array
	#=============================
	def read_track_as_1darray(self, file_name):
		lines = list()
		with open(file_name) as track_file:
			for line in track_file:
				lines.append(line)
		lines = lines[1:] #account for first line being size
		return lines

	#=============================
	# read_track_as_2darray()
	#	- read the track from a given file
	#@param		file_name
	#@return	data as 2d array
	#=============================
	def read_track_as_2darray(self, file_name):
		lines = list()
		with open(file_name) as track_file:
			for line in track_file:
				tmpline = list()
				for char in line:
					tmpline.append(char)
				tmpline = tmpline[:-1] #remove the \n
				lines.append(tmpline)
		lines = lines[1:] #account for first line being dimensions of grid
		return lines

	#=============================
	# find_starting_points()
	#	- find the end starting points of the track
	#	- assumes starting points are in a straight line
	#@return	list of starting points as tuples
	#=============================
	def find_starting_points(self, track):
		#Search for character 'S'
		starting_points = [(ix,iy) for ix, row in enumerate(track) for iy, i in enumerate(row) if i == self.START_CHAR]
		return starting_points

	#=============================
	# find_finish_points()
	#	- find the finish points of the track
	#	- assumes finish points rae in a straight line
	#@return	list of starting points as tuples
	#=============================
	def find_finish_points(self, track):
		#Search for character 'F'
		finish_points = [(ix,iy) for ix, row in enumerate(track) for iy, i in enumerate(row) if i == self.END_CHAR]
		return finish_points

	#=============================
	# find_finish_line()
	#	- find the line/boundaries which define the finish line
	#	- assumes finish points rae in a straight line
	#@return	tuple	(type_of_line, pt1, pt2) where type of line is horizontal (0) or vert(1)
	#=============================
	def find_finish_line(self):

		max_diff = 0
		max_diff_pt1 = 0
		max_diff_pt2 = 0

		for point_idx1 in range(0, len(self.finish_points)):
			test_point1 = np.array(self.finish_points[point_idx1])
			for point_idx2 in range(0, len(self.finish_points)):
				test_point2 = np.array(self.finish_points[point_idx2])
				diff = np.sum(np.abs(test_point1 - test_point2))
				if diff > max_diff:
					max_diff_pt1 = test_point1
					max_diff_pt2 = test_point2
					max_diff = diff

		type_of_line = 0 #horizontal(0) or vertical(1)
		difference = max_diff_pt1 - max_diff_pt2
		if difference[0] == 0:
			type_of_line = 0 #horizontal
		else:
			type_of_line = 1 #vertical

		return (type_of_line, max_diff_pt1, max_diff_pt2)
	
	#=============================
	# find_wall_points()
	#	- find the wall points of the track
	#@return	list of starting points as tuples
	#=============================
	def find_wall_points(self, track):
		#Search for character '#'
		wall_points = [(ix,iy) for ix, row in enumerate(track) for iy, i in enumerate(row) if i == self.WALL_CHAR]
		return wall_points

	#=============================
	# is_wall_point()
	#	- determine if position is wall 
	#@param	position	(x,y)
	#@return	bool result
	#=============================
	def is_wall_point(self, position):
		if position[0] > (self.shape[0]-1) or position[1] > (self.shape[1]-1):
			return True
		elif(position[0] < 0 or position[1] < 0):
			return True
		elif(self.data[position[0]][position[1]] == self.WALL_CHAR):
			return True
		else:
			return False

	#=============================
	# is_track_point()
	#	- determine if position is track 
	#@param	position	(x,y)
	#@return	bool result
	#=============================
	def is_track_point(self, position):
		if (self.data[position[0], position[1]] == self.TRK_CHAR):
			return True
		else:
			return False

	#=============================
	# is_start_point()
	#	- determine if position is start 
	#@param	position	(x,y)
	#@return	bool result
	#=============================
	def is_start_point(self, position):
		if (self.data[position[0], position[1]] == self.START_CHAR):
			return True
		else:
			return False

	#=============================
	# is_track_point()
	#	- determine if position is track 
	#@param	position	(x,y)
	#@return	bool result
	#=============================
	def is_track_point(self, position):
		if (self.data[position[0], position[1]] == self.track_char):
			return True
		else:
			return False

	#=============================
	# find_track_points()
	#	- find the track points of the track
	#@return	list of starting points as tuples
	#=============================
	def find_track_points(self, track):
		#Search for character track_char
		track_points = [(ix,iy) for ix, row in enumerate(track) for iy, i in enumerate(row) if i == self.TRK_CHAR]
		return track_points

	#=============================
	# find_closest_track_point()
	#	- find the track point which is closest 
	#@return	list of starting points as tuples
	#=============================
	def find_closest_track_point(self, position):
		pos = np.array(position)
		track_points = np.array(self.track_points)
		difference_vals = np.sum(np.abs(track_points - pos), axis=1)
		closest_idx = np.argmin(difference_vals)
		closest_pos = self.track_points[closest_idx]
		return closest_pos

	#=============================
	# find_closest_starting_point()
	#	- find the starting points of the track
	#@return	list of starting points as tuples
	#=============================
	def find_closest_starting_point(self, position):
		pos = np.array(position)
		starting_points = np.array(self.start_points)
		difference_vals = np.sum(np.abs(starting_points - pos), axis=1)
		closest_idx = np.argmin(difference_vals)
		closest_pos = self.start_points[closest_idx]
		return closest_pos

	#=============================
	# check_finish_line()
	#	- are you on, or did you cross the finish line?
	#	- Based on the premise the finish line is in fact a line in the x or y dim
	#	- Using this fact, finds the value range to check against position
	#@param		position	position to chec
	#@return	TRUE if at or beyond on the finish line, otherwise false
	#=============================
	def check_finish_line(self, position1, position2):
		#TODO: implement
		crossed_finish_line = False

		position1 = np.array(position1)
		position2 = np.array(position2)

		#For now, just check that you're close-to the line & cross it 
		#Check to see if you're within the bounds & cross
		if self.finish_line[0] == 0: #horizontal line
			#Check X value pos1/2 diff w/ line to be different signs i.e. they crossed (or 0)
			x1 = self.finish_line[1][0]
			xp1 = position1[0]
			xp1_diff = x1 - xp1

			xp2 = position2[0]
			xp2_diff = x1 - xp2
			x_bounds_check = (xp1_diff < 0 < xp2_diff) or (xp2_diff < 0 < xp1_diff) or (xp2_diff == 0)#in case you landed on the line!

			#Check that Y value position1 is within bounds
			y1 = self.finish_line[1][1]
			y2 = self.finish_line[2][1]
			yp1 = position1[1]
			y_bounds_check = (y1 <= yp1 <= y2) or (y2 <= yp1 <= y1) 

			crossed_finish_line = x_bounds_check and y_bounds_check

		else: #vertical line
			#Check Y value pos1/2 diff w/ line to be different signs i.e. they crossed (or 0)
			y1 = self.finish_line[1][1]
			yp1 = position1[1]
			yp1_diff = y1 - yp1

			yp2 = position2[1]
			yp2_diff = y1 - yp2
			y_bounds_check = (yp1_diff < 0 < yp2_diff) or (yp2_diff < 0 < yp1_diff) or (yp2_diff == 0)#in case you landed on the line!

			#Check that Y value position1 is within bounds
			x1 = self.finish_line[1][0]
			x2 = self.finish_line[2][0]
			xp1 = position1[1]
			x_bounds_check = (x1 <= xp1 <= x2) or (x2 <= xp1 <= x1)

			crossed_finish_line = x_bounds_check and y_bounds_check

		return crossed_finish_line


#=============================
# MAIN PROGRAM
#=============================
def main():
	print()
	#print('Main() - testing track object')
	print()
	parser = argparse.ArgumentParser(description='test track object')
	parser.add_argument('file_name', type=str, help='track file name')
	args = parser.parse_args()
	file_name = args.file_name
	print('INPUT VALUES')
	print('--------------')
	print('file_name: ', file_name)
	print()

	track = Track(file_name)

	print('LOG BASIC DATA')
	print('----------------')
	print('track')
	track.print_track()
	print('track.shape')
	print(track.shape)
	print('track.start_points')
	print(track.start_points)
	print('track.finish_points')
	print(track.finish_points)
	print('track.wall_points')
	print(track.wall_points)
	print('track.track_points')
	print(track.track_points)
	print('track.finish_line')
	print(track.finish_line)
	position = (3,3)
	print('closest trk to pos', position)
	closest_pos = track.find_closest_track_point(position)
	print(closest_pos)
	print('closest start to pos', position)
	closest_pos = track.find_closest_starting_point(position)
	print(closest_pos)
	p1 = (2,4)
	p2 = (0,4)
	print('did positions cross finish line', p1, p2)
	crossed_line = track.check_finish_line(p1,p2)
	print(crossed_line)
	print()


if __name__ == "__main__":
	main()

