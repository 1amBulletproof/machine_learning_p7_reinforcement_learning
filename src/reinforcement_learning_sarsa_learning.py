#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			12/5/2018
#@description	Reinforcement Learning w/ SARSA-learning Iteration

import numpy as np
import pandas as pd
import argparse
import operator
import random
import copy
from reinforcement_learning_q_learning import ReinforcementLearningQLearning
from track import Track
from car import Car

#=============================
# ReinforcementLearningSarsaLearning
#
# - Class to encapsulate a reinforcement learning sarsa-learning iteration model
#=============================
class ReinforcementLearningSarsaLearning(ReinforcementLearningQLearning) :

	#=============================
	# __init__()
	#	- create track, super class constructors, and create initial model states & values
	#=============================
	def __init__(self, file_name):
		ReinforcementLearningQLearning.__init__(self, file_name)

	#=============================
	# train()
	#	- OVERRIDED from ReinforcementLearningQLearning - different update algo
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

			#SARSA diff
			#Get action 'a' to take via epsilon greedy algorithm
			action_vals = self.q_table[x_idx][y_idx][x_vel_idx][y_vel_idx]
			action = self.epsilon_greedy_action_choice(epsilon_value, action_vals) #This is the acceleration to choose
			q_val = action_vals[action[0]][action[1]]

			finish_line = False;
			test_steps = 0
			max_test_steps = 999
			while (not finish_line and test_steps < max_test_steps):
				#Keep track of test steps
				test_steps += 1
				if test_steps > max_test_steps:
					self.converge_result.append(False)

				#TAKE ACTION 'A' Get next state via applying action
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
					x_idx_next = car.position[0]
					y_idx_next = car.position[1]
					x_vel_idx_next = car.velocity[0] + self.velocity_offset #account for offset to make index positive
					y_vel_idx_next = car.velocity[1] + self.velocity_offset #account for offset to make index positive

					#Get action 'a' to take via epsilon greedy algorithm
					action_vals_next = self.q_table[x_idx_next][y_idx_next][x_vel_idx_next][y_vel_idx_next]
					action_next = self.epsilon_greedy_action_choice(epsilon_value, action_vals_next) #This is the acceleration to choose
					q_val_next = action_vals_next[action_next[0]][action_next[1]]

					#Q-learning equation! 
					action_vals[action[0]][action[1]] += learning_rate * \
							(reward + discount_factor * q_val_next - q_val)

					#Update for next round
					action_vals = action_vals_next
					action = action_next
					q_val = q_val_next 

			self.history_of_learning.append(test_steps) #track how many steps taken

		return (self.history_of_learning, self.converge_result)


#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing SARSA-learning iteration')
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
	sarsa_learning = ReinforcementLearningSarsaLearning(track_file)
	learn_result = sarsa_learning.train(num_iterations, crash_algo)

	print()
	test_result = sarsa_learning.test(crash_algo)

	if learning_analysis != 0:
		print()
		print('learning analysis')
		print('Training file:', track_file, 'for', num_iterations, ' and crash algo:', crash_algo)
		print('Train Iteration, Steps required')
		for idx in range(0, len(learn_result[0])):
			print(idx, learn_result[0][idx])


if __name__ == '__main__':
	main()
	
