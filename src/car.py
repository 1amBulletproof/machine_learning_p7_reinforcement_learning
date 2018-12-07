#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			12/5/2018
#@description	car

from track import Track
import argparse

#=============================
# Car
#
# - Class to encapsulate car data
#=============================
class Car():
	#=============================
	# __init__()
	#	- constructor
	#	- Crash_type = 0 for minor, 1 for major
	#=============================
	def __init__(self, track, position=[0,0], velocity=[0,0], crash_type=0):
		self.track = track
		self.position = position
		self.previous_position = position
		self.velocity = velocity
		self.velocity_limit = 5
		if (crash_type == 0):
			self.crash = self.minor_crash
		else:
			self.crash = self.major_crash

	#=============================
	# accelerate()
	#	- apply acceleration (effectivley change velocity)
	#@param		acceleration	(x,y) values
	#=============================
	def accelerate(self, acceleration):
		acceleration_limit = 1
		#TODO: apply 20% failure rate
		if abs(acceleration[0]) > acceleration_limit or abs(acceleration[1]) > acceleration_limit:
			print('bad acceleration:', acceleration)
			return

		new_x_velocity = self.velocity[0] + acceleration[0]
		new_y_velocity = self.velocity[1] + acceleration[1]

		if (abs(new_x_velocity) <= self.velocity_limit):
			self.velocity[0] = new_x_velocity
		if (abs(new_y_velocity) <= self.velocity_limit):
			self.velocity[1] = new_y_velocity

		return self.velocity
	
	#=============================
	# move()
	#	- apply velocity to position, check for walls & finish line
	#@return	crossed finish line True, False
	#=============================
	def move(self):
		self.previous_position = self.position
		next_position = [self.position[0] + self.velocity[0], self.position[1] + self.velocity[1]]

		if (self.track.check_finish_line(self.position, next_position)):
			#Crossed the finish line!
			return True
		elif (self.will_crash(next_position)):
			#print('crashing')
			#Handle crash
			self.crash()
		else:
			self.position = [self.position[0] + self.velocity[0], self.position[1] + self.velocity[1]]
		return False

	#=============================
	# will_crash()
	#	- check if car position is wall position
	#@param	position	position to check
	#@return	True or False
	#=============================
	def will_crash(self, position):
		return self.track.is_wall_point(position)

	#=============================
	# minor_crash()
	#	- position = closest safe (track) state
	#	- velocity = (0,0)
	#@return	car with new position & velocity
	#=============================
	def minor_crash(self):
		self.velocity = [0,0]
		self.position = self.previous_position
		#ALTERNATIVE
		#Find closest safe position on track
		#closest_trk_pos = self.track.find_closest_track(self.position)
		return self.position

	#=============================
	# major_crash()
	#	- position = starting_point
	#	- velocity = (0,0)
	#@return	car with new position & velocity
	#=============================
	def major_crash(self):
		self.velocity = [0,0]
		self.position = self.track.find_closest_starting_point(self.position)
		return self.position


#=============================
# MAIN PROGRAM
#=============================
def main():
	print()
	#print('Main() - testing car object')
	print()
	parser = argparse.ArgumentParser(description='test car object')
	parser.add_argument('file_name', type=str, help='track file name')
	args = parser.parse_args()
	file_name = args.file_name
	print('INPUT VALUES')
	print('--------------')
	print('file_name: ', file_name)
	print()

	track = Track(file_name)
	print('Making car1 w/ minor_crash & car2 w/ major_crash')
	car1 = Car(track, track.start_points[0])
	car2 = Car(track, track.start_points[0], [0,0], 1) #severe crash

	print('LOG BASIC DATA')
	print('----------------')
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	print('accelerate (0,1)')
	car1.accelerate((0,1))
	car2.accelerate((0,1))

	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)
	car1.move()
	car2.move()
	print('car1 vel', car1.velocity)
	print('car1 pos', car1.position)
	print('car2 vel', car2.velocity)
	print('car2 pos', car2.position)



if __name__ == "__main__":
	main()


	

