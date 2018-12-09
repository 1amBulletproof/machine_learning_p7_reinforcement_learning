#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			12/5/2018
#@description	Reinforcement Learning w/ Value Iteration

import numpy as np
import pandas as pd
import argparse
import operator
import random
import copy
from base_model2 import BaseModel
from race_simulator import RaceSimulator
from track import Track
from car import Car

#=============================
# ReinforcementLearningValueIteration
#
# - Class to encapsulate a reinforcement learning value iteration model
#=============================
class ReinforcementLearningValueIteration(BaseModel, RaceSimulator):

	#=============================
	# __init__()
	#	- create track, super class constructors, and create initial model states & values
	#=============================
	def __init__(self, file_name):
		new_track = Track(file_name)
		BaseModel.__init__(self, new_track.data)
		RaceSimulator.__init__(self, new_track)
		self.accelerations = [[-1,-1],[-1,0],[-1,1], [0,-1],[0,0],[0,1], [1,-1],[1,0],[1,1]]
		self.velocity_range = 11 # This is {+/-5} offset from 0
		self.velocity_offset = 5
		self.accel_range = 3 # This is {+/-1} offset from 0
		self.accel_offset = 1
		self.v_table = self.create_v_table(self.track.shape) #state value table
		self.p_table = self.create_v_table(self.track.shape) #Policy table
		self.q_table = self.create_q_table(self.track.shape) #q_table values

	#=============================
	# create_q_table()
	#	- create the multi-dimensional table representing the state space
	#	- NOTE: velocities will be offset to make indexing easier, -5:0, -4:1, -3:2 ... 0:5, 1:6, 2:7, ... 5:10
	#		- simply add '5' to the velocity value to reach it's index
	#=============================
	def create_q_table(self, grid_shape):
		state_action_tbl = list()
		for x in range(grid_shape[0]):
			x_list = list()
			for y in range(grid_shape[1]):
				y_list = list()
				for x_vel in range(0, self.velocity_range):
					x_vel_list = list()
					for y_vel in range(0, self.velocity_range):
						#initialize state value to 0
						y_vel_list = list()
						for idx in range(0, len(self.accelerations)):
							y_vel_list.append(0)
						x_vel_list.append(y_vel_list)
					y_list.append(x_vel_list)
				x_list.append(y_list)
			state_action_tbl.append(x_list)
		return state_action_tbl

	#=============================
	# create_v_table()
	#	- create the multi-dimensional table representing the state space
	#	- NOTE: velocities will be offset to make indexing easier, -5:0, -4:1, -3:2 ... 0:5, 1:6, 2:7, ... 5:10
	#		- simply add '5' to the velocity value to reach it's index
	#=============================
	def create_v_table(self, grid_shape):
		state_value_table = list()
		for x in range(grid_shape[0]):
			x_list = list()
			for y in range(grid_shape[1]):
				y_list = list()
				for x_vel in range(0, self.velocity_range):
					x_vel_list = list()
					for y_vel in range(0, self.velocity_range):
						#initialize state value to 0
						x_vel_list.append(0)
					y_list.append(x_vel_list)
				x_list.append(y_list)
			state_value_table.append(x_list)
		return state_value_table

	#=============================
	# train()
	#	- Learn Values of every state (via value iteration)
	#=============================
	def train(self, max_iterations, car_algo):
		#initialize values
		tmp_car = self.create_start_car(car_algo)
		discount_factor = 0.95
		reward = -1
		bellman_error_magnitude = 0.1
		self.training_iterations = 0
		max_delta = 0
		error_history = [max_delta]

		done = False
		while(not done and self.training_iterations < max_iterations):
			max_delta = 0
			self.training_iterations += 1
			#Make previous deep copy
			v_table_previous = copy.deepcopy(self.v_table)

			#Iterate through & update EVERY state
			for x in range(0, len(self.v_table)):
				y_vals = self.v_table[x]
				for y in range(0, len(y_vals)):
					x_vel_vals = self.v_table[x][y]
					for x_vel in range(0, len(x_vel_vals)):
						y_vel_vals = self.v_table[x][y][x_vel]
						for y_vel in range(0, len(y_vel_vals)):
							#UPDATE THIS STATE VALUE & POLICY
							max_q_val = -999999
							policy = [0,0]
							for accel_idx in range(0, len(self.accelerations)):
								acceleration = self.accelerations[accel_idx]
								reward = -1
								#Find the next state
								tmp_car.position = [x,y]
								tmp_car.velocity = [x_vel - self.velocity_offset, y_vel - self.velocity_offset]
								tmp_car.accelerate(acceleration)

								cross_finish_line = tmp_car.move()

								#find the next state value
								x_next = tmp_car.position[0] #Note if crossing finish line, DO NOT USE, position not updated
								y_next = tmp_car.position[1]
								x_next_vel = tmp_car.velocity[0] + self.velocity_offset
								y_next_vel = tmp_car.velocity[1] + self.velocity_offset
								
								state_val_next = float(0)
								if cross_finish_line:
									state_val_next = float(0)
									reward = 0
								else:
									state_val_next = float(v_table_previous[x_next][y_next][x_next_vel][y_next_vel])

								#Calculate q_value 
								q_val = reward + discount_factor * state_val_next
								self.q_table[x][y][x_vel][y_vel][accel_idx] = q_val

								#Track the max q_val
								if q_val > max_q_val:
									policy = acceleration
									max_q_val = q_val

							#update the value & policy w/ the best action result
							#The value should be the max q-value
							prev_q_val = self.v_table[x][y][x_vel][y_vel]
							self.v_table[x][y][x_vel][y_vel] = max_q_val
							#Remember the max delta for stopping point
							delta = prev_q_val - max_q_val
							if delta > max_delta:
								print('itr:', self.training_iterations, 'new max value delta', delta)
								max_delta = delta
							#The action associated with this q-value is now the policy
							self.p_table[x][y][x_vel][y_vel] = policy
			error_history.append(max_delta)
			if max_delta < bellman_error_magnitude:
				done = True

		return (self.training_iterations, error_history)

	#=============================
	# test()
	#	- test the model (i.e. traverse using policy p_table) 
	#	- Uses the given training data, otherwise wont work
	#@return				value of performance
	#=============================
	def test(self, crash_algo):
		#Initialize car to random location
		test_car = self.create_start_car(crash_algo)
		self.history.append(test_car.position)
		self.iterations = 0

		crossed_finish = False
		while(not crossed_finish and self.iterations < 50):
			#self.print_state(test_car)
			#get proper indexing for state-action (q-table) 
			x_idx = test_car.position[0]
			y_idx = test_car.position[1]
			x_vel_idx = test_car.velocity[0] + self.velocity_offset #account for offset to make index positive
			y_vel_idx = test_car.velocity[1] + self.velocity_offset #account for offset to make index positive

			#Get action 'a' acceleration to take via policy
			acceleration = self.p_table[x_idx][y_idx][x_vel_idx][y_vel_idx]

			#Get next state via applying action
			test_car.accelerate(acceleration)
			done = self.move(test_car)
			if done:
				crossed_finish = True

		self.print_state(test_car)
		self.print_history()
		return (self.iterations, self.history)

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing value iteration')
	parser = argparse.ArgumentParser(description='test value iteration')
	parser.add_argument('track_file', type=str, default=1, help='track file name')
	parser.add_argument('max_iterations', type=int, default=999, help='max number of iterations')
	parser.add_argument('crash_algorithm', type=int, default=0, help='crash algo: 0 = minor, 1 = major')
	parser.add_argument('learning_analysis', type=int, default=0, help='do learning analysis or not, 0 = no, 1 = yes')
	args = parser.parse_args()

	track_file = args.track_file
	max_iterations = args.max_iterations
	crash_algo = args.crash_algorithm
	learning_analysis = args.learning_analysis

	print()
	print('Training file:', track_file, 'for max itr', max_iterations, ' and crash algo:', crash_algo)

	print()
	value_iteration = ReinforcementLearningValueIteration(track_file)
	learn_result = value_iteration.train(max_iterations, crash_algo)
	print('Training results')
	print('training iterations', learn_result)

	print()
	test_result = value_iteration.test(crash_algo)

	if learning_analysis != 0:
		print()
		print('learning analysis')
		print('Training file:', track_file, 'for max iter', max_iterations, ' and crash algo:', crash_algo)
		print('Train Itr, Max Error')
		for idx in range(0, learn_result[0]):
			print(idx, learn_result[1][idx])


if __name__ == '__main__':
	main()
	
