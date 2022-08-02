# Import routines

import numpy as np
import math
import random

from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2))
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encoding = [0 for _ in range(m+t+d)]

        state_encoding[state[0]] = 1
        state_encoding[m+state[1]] = 1
        state_encoding[m+t+state[2]] = 1

        return state_encoding


#     Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
#         """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        
#         state_encoding = [0 for _ in range(m+t+d+m+m)]

#         state_encoding[state[0]] = 1
#         state_encoding[m+state[1]] = 1
#         state_encoding[m+t+state[2]] = 1

#         if action[0] != 0:
#             state_encoding[m+t+d+action[0]] = 1 
#         if action[1] != 0:
#             state_encoding[m+t+d+m+action[1]] = 1 
        
        
#         return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_idx = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        actions.append((0,0))
        possible_actions_idx.append(0)

        return possible_actions_idx,actions   


    ## Making the reward functions

    def reward_func(self, offline_time, transit_time, ride_time):
        """Takes in time variables and returns the reward"""
        
        revenue_time = ride_time
        
        non_revenue_time = offline_time + transit_time
        
        reward = (R * revenue_time) - (C * ( revenue_time + non_revenue_time ))
        
        return reward

    ## Making the step function

    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # Get the next state and the various time durations
        next_state, offline_time, transit_time, ride_time = self.next_state_func(
            state, action, Time_matrix)

        # Calculate the reward based on the different time durations
        rewards = self.reward_func(offline_time, transit_time, ride_time)
        total_time = offline_time + transit_time + ride_time
        
        return rewards, next_state, total_time


    ## Getting the next state
    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state as well as the time variables"""
        
        #Initialising time variables
        
        total_time = 0
        transit_time = 0 # from curr location to pickup point
        ride_time = 0 #from pickup to drop point
        offline_time = 0 #driver goes offline
        
        cur_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        cur_time = state[1]
        cur_day = state[2]
        
        if ((pickup_loc == 0) & (drop_loc == 0)): ## case when driver goes offline
            offline_time = 1
            new_loc = cur_loc
            
        elif (cur_loc == pickup_loc): # case when the pickup location is the current location
            ride_time = Time_matrix[cur_loc][drop_loc][cur_time][cur_day]
            new_loc = drop_loc
            
        else: # when driver is not at the pickup location
            transit_time = Time_matrix[cur_loc][pickup_loc][cur_time][cur_day]
            
            new_time, new_day = self.update_time_day(cur_time, cur_day, transit_time)
            
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            new_loc  = drop_loc
            
         # Calculate total time as sum of all durations
        total_time = (offline_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(cur_time, cur_day, total_time)
        
        next_state = [new_loc,next_time,next_day]
            
        return next_state,offline_time,transit_time,ride_time


    ##  Returning the action and state variables
    
    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    
    ## Returning updated time and day
    
    def update_time_day(self, time, day, ride_duration):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            time = time + ride_duration
        else:
            time = (time + ride_duration) % 24 
            
            num_days = (time + ride_duration) // 24
            
            day = (day + num_days ) % 7

        return time, day
