#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	RaceSimulator class

from track import Track
from car import Car
import numpy as np
import pandas as pd
import argparse
import operator
import copy
import random
import os
import copy

#=============================
# RaceSimulator
#
# - Base class of class to emulate racing on the track
#=============================
class RaceSimulator:

	def __init__(self, track, car_symbol='@'):
		self.track = track
		#Keep track of the board
		self.display_track = copy.copy(track.data)
		self.prev_char_removed = 'S' #assumes you start on a start line
		self.car_symbol = car_symbol
		#Keep track of game status
		self.game_state = np.zeros(track.shape)
		self.iterations = 0
		self.history = list()

	#=============================
	# print_state()
	#	- print the state of the game
	#=============================
	def print_state(self, car):
		os.system('clear')
		#Game state
		print('iterations: ', self.iterations)
		print('history: ', self.history)
		print('velocity: ', car.velocity)
		#Game trk
		for lines in self.display_track:
			print(''.join(str(n) for n in lines))

	#=============================
	# print_history()
	#	- print the state of the game
	#=============================
	def print_history(self):
		print()
		print('HISTORY')
		print('-------')
		for idx in range(0, len(self.history)):
			position = self.history[idx]
			self.display_track[position[0]][position[1]] = idx
		for lines in self.display_track:
			print(''.join(str(n) for n in lines))

	#=============================
	# move()
	#	- update the game for a move
	#	- check to see if the game is over
	#=============================
	def move(self, car):
		crossed_finish = car.move()
		if (crossed_finish):
			#Crossed the finish line!
			print('Passed finish line!')
			return True

		self.iterations += 1 #we will not count the winning move
		self.history.append(car.position)
		car_prev = car.previous_position
		car_pos = car.position

		if (car_prev == car_pos):
			#Car didn't change positions
			return
		else: #new car position, fill in the old one and rememer the displaced value
			self.display_track[car_prev[0]][car_prev[1]] = self.prev_char_removed
			self.prev_char_removed  = self.display_track[car_pos[0]][car_pos[1]]
			self.display_track[car_pos[0]][car_pos[1]] = self.car_symbol

		return False

	#=============================
	# create_start_car()
	#	- initialize car
	#@param	crash_algo 0 = minor crash, 1 = major crash
	#=============================
	def create_start_car(self, crash_algo):
		#Get a random starting point
		num_start_pos = len(self.track.start_points)
		start_pos = random.randrange(num_start_pos)
		start_pt = self.track.start_points[start_pos]
		init_velocity = [0,0]
		car = Car(self.track, start_pt, init_velocity, crash_algo)
		return car

	#=============================
	# race()
	#	- race a car through a track
	#=============================
	def race(self, crash_algo ):
		car = self.create_start_car(crash_algo)

		start_pt = car.position 
		self.prev_char_removed = self.display_track[start_pt[0]][start_pt[1]]
		self.display_track[start_pt[0]][start_pt[1]] = self.car_symbol
		self.iterations = 0
		self.history.append(car.position)

		done = False
		while(not done):
			self.print_state(car)
			#handle input
			key = input('space to move, wasd to accel ')
			if key == ' ':
				done = self.move(car)
			elif key == 'w':
				accel = (-1,0)
				car.accelerate(accel)
			elif key == 's':
				accel = (1,0)
				car.accelerate(accel)
			elif key == 'a':
				accel = (0,-1)
				car.accelerate(accel)
			elif key == 'd':
				accel = (0,1)
				car.accelerate(accel)
			else:
				print('wrong key, use wasd and space')

		self.print_state(car)
		self.print_history()
		return (self.iterations, self.history)

	
#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing base race simulator')
	print()
	parser = argparse.ArgumentParser(description='test track object')
	parser.add_argument('file_name', type=str, help='track file name')
	parser.add_argument('crash_algo', type=int, help='crash algorithm, 0 or 1')
	args = parser.parse_args()
	file_name = args.file_name
	crash_algo = args.crash_algo
	print('INPUT VALUES')
	print('--------------')
	print('file_name: ', file_name)
	print('crash_algo: ', crash_algo)
	print()

	track = Track(file_name)
	race_sim = RaceSimulator(track)
	race_sim.race(crash_algo)



if __name__ == '__main__':
	main()
