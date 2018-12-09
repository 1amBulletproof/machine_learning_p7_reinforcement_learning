#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			12/5/2018
#@description	Reinforcement Learning w/ Q-learning Iteration

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
# ReinforcementLearningQLearning
#
# - Class to encapsulate a reinforcement learning q-learning iteration model
#=============================
class ReinforcementLearningQLearning(BaseModel, RaceSimulator ) :

	#=============================
	# __init__()
	#	- create track, super class constructors, and create initial model states & values
	#=============================
	def __init__(self, file_name):
		new_track = Track(file_name)
		BaseModel.__init__(self, new_track.data)
		RaceSimulator.__init__(self, new_track)
		#acceleration possible [(-1,-1),(-1,0),(-1,1), (0,-1),(0,0),(0,1), (1,-1),(1,0),(1,1)]
		self.velocity_range = 11 # This is {+/-5} offset from 0
		self.velocity_offset = 5
		self.accel_range = 3 # This is {+/-1} offset from 0
		self.accel_offset = 1
		self.q_table = self.create_q_table(self.track.shape) 

	#=============================
	# create_q_table()
	#	- create the multi-dimensional table representing the state space
	#	- NOTE: velocities will be offset to make indexing easier, -5:0, -4:1, -3:2 ... 0:5, 1:6, 2:7, ... 5:10
	#		- simply add '5' to the velocity value to reach it's index
	#	- NOTE: acceleration will be offset to make indexing easier, -1:0, 0:1, 1:2
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
						#initialize accelerateion reward value (offset like velocity)
						random_init_vals = np.random.rand(self.accel_range, self.accel_range)
						random_init_vals *= -1 #since everything is negative,  might as well
						x_vel_list.append(random_init_vals)
					y_list.append(x_vel_list)
				x_list.append(y_list)
			state_action_tbl.append(x_list)
		return state_action_tbl

	#=============================
	# epsilon_greedy_action_choice()
	#	- epsilon greedy approach
	#@return index of selected action
	#=============================
	def epsilon_greedy_action_choice(self, epsilon, actions):
		if np.random.random() < epsilon: 
			#either this is the initial pass or we chose to select randomly
			action = (np.random.randint(0,self.accel_range), np.random.randint(0,self.accel_range))
		else: #get the max value
			raw_index = np.where(actions.max() == actions)
			action = (raw_index[0][0], raw_index[1][0])
		return action

	#=============================
	# create_random_car()
	#	- initialize car
	#@param	crash_algo 0 = minor crash, 1 = major crash
	#=============================
	def create_random_car(self, crash_algo):
		#Get a random starting point
		num_start_pos = len(self.track.valid_points)
		start_pos = random.randrange(num_start_pos)
		start_pt = self.track.valid_points[start_pos]
		init_velocity = [0,0]
		car = Car(self.track, start_pt, init_velocity, crash_algo)
		return car

	#=============================
	# train()
	#	- Learn Values of every state (via q-learning)
	#=============================
	def train(self, number_of_iterations, crash_algo):
		#initialize values
		learning_rate = 0.75
		min_learning_rate = 0.01
		discount_factor = 0.95
		reward = -1
		epsilon_value = 0.5 #half the time we explore, the other half we exploit
		decay = 0.9999

		self.history_of_learning = list() #store number of test_steps taken per iteration
		self.converge_result = list() #store whether it converged

		for idx in range(number_of_iterations):
			#init the environment, aka put the car at a starting place w/ 0 velocity
			car = self.create_start_car(crash_algo)

			#Reduce learning & epsilon value over time to explore less and learn less
			if learning_rate > min_learning_rate:
				learning_rate *= decay
			epsilon_value *= decay

			test_steps = 0

			#get proper indexing for state-action (q-table) 
			x_idx = car.position[0]
			y_idx = car.position[1]
			x_vel_idx = car.velocity[0] + self.velocity_offset #account for offset to make index positive
			y_vel_idx = car.velocity[1] + self.velocity_offset #account for offset to make index positive

			finish_line = False;
			test_steps = 0
			max_test_steps = 999
			while (not finish_line and test_steps < max_test_steps):
				#Keep track of test steps
				test_steps += 1
				if test_steps > max_test_steps:
					self.converge_result.append(False)

				#Get action 'a' to take via epsilon greedy algorithm
				action_vals = self.q_table[x_idx][y_idx][x_vel_idx][y_vel_idx]
				action = self.epsilon_greedy_action_choice(epsilon_value, action_vals) #This is the acceleration to choose
				q_val = action_vals[action[0]][action[1]]

				#Get next state via applying action
				acceleration = [action[0] - self.accel_offset, action[1] - self.accel_offset]
				car.accelerate(acceleration)
				done  =  car.move()
				if done:
					#We are finished, 
					finish_line = True
					self.converge_result.append(True)
				else:

					#Get next action values
					#get proper indexing for state-action (q-table) 
					x_idx = car.position[0]
					y_idx = car.position[1]
					x_vel_idx = car.velocity[0] + self.velocity_offset #account for offset to make index positive
					y_vel_idx = car.velocity[1] + self.velocity_offset #account for offset to make index positive
					action_vals_next = self.q_table[x_idx][y_idx][x_vel_idx][y_vel_idx]
					max_q_val_next = np.max(action_vals_next)

					#Q-learning equation! 
					action_vals[action[0]][action[1]] += learning_rate * \
							(reward + discount_factor * max_q_val_next - q_val)

			self.history_of_learning.append(test_steps) #track how many steps taken

		return (self.history_of_learning, self.converge_result)

	#=============================
	# test()
	#
	#	- test the model (i.e. traverse using policy (maxQ value)
	#@return				value of performance
	#=============================
	def test(self, crash_algorithm):
		self.iterations = 0
		epsilon_value = 0 #Will allow use epsilon greedy algo to get 100% exploitation action

		#Initialize car to random location
		car = self.create_start_car(0)
		self.history.append(car.position)

		crossed_finish = False
		while(not crossed_finish and self.iterations < 999):
			self.print_state(car)
			#get proper indexing for state-action (q-table) 
			x_idx = car.position[0]
			y_idx = car.position[1]
			x_vel_idx = car.velocity[0] + self.velocity_offset #account for offset to make index positive
			y_vel_idx = car.velocity[1] + self.velocity_offset #account for offset to make index positive

			#Get action 'a' to take via epsilon greedy algorithm
			action_vals = self.q_table[x_idx][y_idx][x_vel_idx][y_vel_idx]
			action = self.epsilon_greedy_action_choice(epsilon_value, action_vals) #This is the acceleration to choose
			acceleration = [action[0] - self.accel_offset, action[1] - self.accel_offset]
			#Get next state via applying action
			car.accelerate(acceleration)
			done = self.move(car)
			if done:
				crossed_finish = True

		self.print_state(car)
		self.print_history()
		return (self.iterations, self.history)

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing q-learning iteration')
	parser = argparse.ArgumentParser(description='test value iteration')
	parser.add_argument('track_file', type=str, default=1, help='track file name')
	parser.add_argument('number_of_iterations', type=int, default=999, help='number of iterations')
	parser.add_argument('crash_algorithm', type=int, default=0, help='crash algo: 0 = minor, 1 = major')
	parser.add_argument('learning_analysis', type=int, default=0, help='do learning analysis or not, 0 = no, 1 = yes')
	args = parser.parse_args()

	track_file = args.track_file
	num_iterations = args.number_of_iterations
	crash_algo = args.crash_algorithm
	learning_analysis = args.learning_analysis

	print()
	print('Training file:', track_file, 'for', num_iterations, ' and crash algo:', crash_algo)

	print()
	q_learning = ReinforcementLearningQLearning(track_file)
	learn_result = q_learning.train(num_iterations, crash_algo)

	print()
	test_result = q_learning.test(crash_algo)

	if learning_analysis != 0:
		print()
		print('learning analysis')
		print('Training file:', track_file, 'for', num_iterations, ' and crash algo:', crash_algo)
		print('Train Iterations, Max delta error')
		for idx in range(0, len(learn_result[0])):
			print(idx, learn_result[0][idx])


if __name__ == '__main__':
	main()
	
