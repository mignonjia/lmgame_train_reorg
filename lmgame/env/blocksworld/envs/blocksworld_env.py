import gym
from gym.spaces import Discrete, Tuple
from gym.utils import seeding
from six import StringIO
import sys
import random
import subprocess
import numpy as np
import copy

#import io
import json
import os

#observation space
#{0,BlocksNum}
#block to move
#{0,BlocksNum -1}
#destination
#{0,BlocksNum}


# IMPORTANT: 
# Before using configure the __init__ function, 
# Configure the variables 'numBlocks' and 'self.bwstates_path'

#The following rewards are considered
#0 when the agent reaches the goal state
#-1 for a valid state
#-10 for reaching a non-valid state (due to the way we code the states some of the combinations are not valid combinations)

DEFAULT_NUM_BLOCKS = 3

class BlocksWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    

    def __init__(self):
        numBlocks = self.readNumBlocksFromFile()
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.bwstates_path = os.path.join(current_path,'BWSTATES/bwstates.1/bwstates')
        #self.bwstates_path = '/home/usuaris/rgarzonj/github/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates'

        self.numBlocks = numBlocks
        self.blocks_ids = [i+1 for i in range(numBlocks)]
        
        # Define action space: [block_to_move, destination]
        # block_to_move: 1 to numBlocks
        # destination: 0 (table) or 1 to numBlocks (on top of another block)
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(numBlocks + 1, start=1),  # 1 to numBlocks
            gym.spaces.Discrete(numBlocks + 1)   # 0 to numBlocks
        ])
        
        # Define observation space: current state + goal state
        obs_size = numBlocks * 2  # State + Goal
        self.observation_space = gym.spaces.Box(
            low=0,
            high=numBlocks,
            shape=(obs_size,),
            dtype=np.int32
        )
        
        self.episode_total_reward = None
        self._seed()
    
    def readNumBlocksFromFile(self):
        numBlocks = DEFAULT_NUM_BLOCKS
        if (os.path.isfile('numBlocks.json')):
             try:
                 to_unicode = unicode
             except NameError:
                 to_unicode = str
                 # Read JSON file
             with open('numBlocks.json') as data_file:
                 data_loaded = json.load(data_file)
             return (data_loaded['numBlocks'])
        else:
            return numBlocks

    
    def generate_random_initial_OLD(self):
        nBlocks = self.numBlocks
        startState = []
        i=1
        while (i<=nBlocks):
            r = random.randint(0,nBlocks)
            print ('i' + str(i))
            print ('r' + str(r))
            while (r!=0 and ((r==i) or (r in startState) or ((r<i) and (startState[r-1]==i)))):
                r = random.randint(0,nBlocks)
                print ('r' + str(r))
            startState.append(r)
            i += 1
            print (str(startState))
        return startState
    
    def generate_random_state (self):
    #""" Generates valid initial state from the implementation of Slaney & Thiébaux""" 
        bwstates_command = self.bwstates_path + ' -n ' + str(self.numBlocks) 
        #+ ' -r ' + str(seed)
        proc = subprocess.Popen(bwstates_command,stdout=subprocess.PIPE,shell=True)
        (out, err) = proc.communicate()
        out_str = out.decode('utf8')
        lines = out_str.split('\n')
        results = lines[1].strip()
        results = results.split()
        res = list(map(int, results))
        return res
    
    def generate_random_goal(self,initialState):
        goalState = self.generate_random_state()
        while (str(initialState)==str(goalState)):
            goalState = self.generate_random_state()
        return (goalState) 
          
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def integer_to_action (self,int_action):
    # From an integer returns the encoded format for an action
    # [block to move, destination]
        ret = []
        ret.append(int(int_action/(self.numBlocks+1)))
        ret.append(int_action%(self.numBlocks+1))
        #print ('\nDecoding action' + str(ret))
        return ret
        
    def action_to_integer (self,action):
    # From an encoded action [block to move, destination] 
    # returns a simple integer
        ret = action[0]*(self.numBlocks-1) + action[1]
        #print ('\Encoding action' + str(ret))
        return ret       

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        
        Args:
            action: The action to take [block_to_move, destination]
            block_to_move: 1 to numBlocks
            destination: 0 (table) or 1 to numBlocks (on top of another block)
            
        Returns:
            tuple:
                - observation (object): Agent's observation of the current environment
                - reward (float): Amount of reward returned after previous action
                - done (bool): Whether the episode has ended
                - info (dict): Contains auxiliary diagnostic information
        """
        block_to_move, destination = action
        
        info = {'success': False, 'valid_move': False}
        
        # Check if the action is valid
        if block_to_move not in self.blocks_ids or (destination not in self.blocks_ids and destination != 0):
            return self.state, -5, False, info
        
        # Check if block is the same as the destination
        if block_to_move == destination:
            return self.state, -5, False, info
        
        # Check if the block to move has any blocks on top of it
        for pos in self.state:
            if pos == block_to_move:  # If any block is on top of the block to move
                return self.state, -5, False, info
                
        # Check if the destination has any blocks on top of it
        if destination != 0:  # If destination is not the table
            for pos in self.state:
                if pos == destination:  # If any block is on top of the destination
                    return self.state, -5, False, info
        
        info['valid_move'] = True
        
        # Apply the move
        new_state = self.state.copy()
        new_state[block_to_move - 1] = destination  # -1 because blocks are 1-indexed
        
        print(f'New state: {new_state}')
        # Calculate reward
        reward = -1  # Default reward for valid action
        done = False
        
        # Check if goal state is reached
        if new_state == self.goal:
            reward = 10
            done = True
            info['success'] = True
            
        # Update state
        self.state = new_state
        
        return self.state, reward, done, info

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation.
        
        Returns:
            tuple:
                - observation (object): The initial observation
                - info (dict): Contains auxiliary diagnostic information
        """
        if seed is not None:
            self._seed(seed)
            
        self.state = self.generate_random_state()
        self.goal = self.generate_random_goal(self.state)

        print(f'State: {self.state}')
        print(f'Goal: {self.goal}')
        return self._get_obs(), {}

    def _get_obs(self):
        print(f'State: {self.state}')
        print(f'Goal: {self.goal}')
        ret = np.concatenate((self.state,self.goal))
        return ret
#        return self.state

    def _render(self, mode='human', close=False):
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("************** New Step ***************\n") 
        outfile.write("[block_to_move, destination]; b [0,numBlocks-1], d[0,numBlocks]\n")                         
        outfile.write("Initial state: " + str(self.initial)+ "\n")
        outfile.write("Current state: " + str(self.state)+ "\n")
#       outfile.write (str(self.state))
        outfile.write("Goal state:    "+ str(self.goal) + "\n")
        outfile.write ("Reward: " + str(self.last_reward)+ "\n")
        outfile.write ("Total Episode Reward: " + str(self.episode_total_reward)+ "\n")
        return outfile

    def plot_row (self,blocksList):
        lines = []
        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + " ¯¯¯ "
        lines.append(oneLine)
        
        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + "| " + str(block+1) + " |"                                            
        lines.append(oneLine)

        oneLine = ""
        for block in blocksList:                
            oneLine = oneLine + " ___ "
        lines.append(oneLine)
        return lines
        
    def _render_v2(self, mode='human', close=False):
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("************** New Step ***************\n") 
        # List the files in the table
        on_table = sorted([i for i, e in enumerate(self.state) if e == 0])
        lines = []
        #lines.append("==============================================================\n")                             
        lines.extend(self.plot_row(on_table))

#        oneLine = ""
#        for block in on_table:                
#            oneLine = oneLine + " ¯¯¯ "
#        lines.append(oneLine)
#        
#        oneLine = ""
#        for block in on_table:                
#            oneLine = oneLine + "| " + str(block+1) + " |"                                            
#        lines.append(oneLine)
#
#        oneLine = ""
#        for block in on_table:                
#            oneLine = oneLine + " ___ "
#        lines.append(oneLine)

        # Take 
        #print ('on_table')
        #print (on_table)
        newrow = []
        on_table = [x+1 for x in on_table]
        print ('on_table')
        print (on_table)
        for block in on_table:                
            #For every block, search if is in the list
            if (block in self.state):
                newrow.append(self.state.index(block))

        while (len(newrow)>0):            
            print('newrow')
            print (newrow)
            lines.extend(self.plot_row(newrow))
            newrow = [x+1 for x in newrow]
            newrow2 = newrow
            newrow = []
            for block in newrow2:                
            #For every block, search if is in the list
                if (block in self.state):
                    newrow.append(self.state.index(block))
            
        # Plot everything in reverse order as it was added

        l = len(lines)-1
        while (l>=0):
            #print (l)
            outfile.write(lines[l])
            outfile.write("\n")
            l = l-1
                        

#        outfile.write("[block_to_move, destination]; b [0,numBlocks-1], d[0,numBlocks]\n")                 
#        outfile.write("Initial state: " + str(self.initial)+ "\n")
#        outfile.write("Current state: " + str(self.state)+ "\n")
#       outfile.write (str(self.state))
#        outfile.write("Goal state:    "+ str(self.goal) + "\n")
#        outfile.write ("Reward: " + str(self.last_reward)+ "\n")
#        outfile.write ("Total Episode Reward: " + str(self.episode_total_reward)+ "\n")
        return outfile

    def seed(self, seed=None):
        """Set the seed for this env's random number generator."""
        return self._seed(seed)
