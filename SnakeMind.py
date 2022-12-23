import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from SnakeLearningRefined import GameEnviroment
from SnakeLearning_RefinedModel import NeuralNetwork, ReinforcementLearning


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
learning_rate = 0.001


class SnakeBrain:

    def __init__(self):
        self.totalgames = 0 # games played
        self.randomness = 0 # randomness
        self.discountrate = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)
        self.MaxRandomNumber = 200
        self.model = NeuralNetwork(16,4) #NN input and output size
        self.RF = ReinforcementLearning(self.model,learning_rate=learning_rate,discount_rate=self.discountrate)

    def train_while_play(self,initial_state,action,reward,next_state,done): # rewards get accounted midway
        self.RF.training(initial_state,action,reward,next_state,done)

    def train_after_play(self): # rewards at the end
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
       
        initial_states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.RF.training(initial_states,actions,rewards,next_states,dones)


    def snake_Action(self,state):

        # as more games are played less randomness over time because it SHOULD learn
        # the randomness allows it to see force to see new actions which might speed 
        # up the learning process
        self.randomness = 80 - self.totalgames 
        #sending an action.
        Action = [0,0,0,0]

        #determines if a random number is done or use NN to complete the action.
        #around after 80 games it should be completly reliant on NN
        if random.randint(0,self.MaxRandomNumber) < self.randomness:
            move = random.randint(0,3)
            Action[move] = 1
        else:
            prediction = self.model(torch.tensor(state,dtype=torch.float))
            move = torch.argmax(prediction).item()
            Action[move] = 1

        return Action

    def remember(self,initial_state,action,reward,next_state,done):
        self.memory.append((initial_state,action,reward,next_state,done))


        #gather states

    def get_State(self,game):

        #it was soon realized that the position updates midway checking states so using the 
        #initial position of the head to get more accurate states
        snakehead = game.snake_position

        checksnakeleft = 0
        checksnakeright = 0
        checksnakeup = 0
        checksnakedown = 0

        CheckRight = 0
        CheckLeft = 0
        CheckUp = 0
        CheckDown = 0

        #print(game.window_x - game.snake_position[0])

        # at any given moment the snake head
        # will compare distances from the border
        # and give an amount it can tell before
        # its dangerous for said snake

        movingleft = 0
        movingright = 0
        movingup = 0
        movingdown = 0


        # Checking the position of windows relative to snake head
        # and calcuating the distance between
        # if it hits a certain threshold then the
        # moving'direction' will return the current
        # state it is wether safe or danger

        if (game.window_x - game.snake_position[0]) > 180:
            CheckLeft = 1

        if (game.window_x - game.snake_position[0]) < 30: 
            CheckRight = 1

        if (game.window_y - game.snake_position[1]) > 180:
            CheckUp = 1

        if(game.window_y - game.snake_position[1]) < 30:
            CheckDown = 1

        # need to get which direction they are moving
        if game.direction == 'UP':
            movingup = 1
        elif game.direction == 'DOWN':
            movingdown = 1
        elif game.direction == 'LEFT':
            movingleft = 1
        elif game.direction == 'RIGHT':
            movingright = 1


            #this fort loop will get the entire snake body and compare it with the head
            #if its anywhere near it in all four directions
        for block in game.snake_body[1:]:
            #print(block)
            if movingright == 1:
                #danger up,down,right
                if ((snakehead[0] >= block[0] - 20) and (block[0] >= snakehead[0])):
                    if(snakehead[1] == block[1]):
                        checksnakeright = 1

                if ((snakehead[1] >= block[1] - 20) and (block[1] >= snakehead[1])):
                    if(snakehead[0] == block[0]):
                        checksnakedown = 1

                if ((snakehead[1] <= block[1] + 20) and (block[1] <= snakehead[1])):
                    if(snakehead[0] == block[0]):
                        checksnakeup = 1

            elif movingleft == 1:
                #danger up,down,left
                if ((snakehead[0] <= block[0] + 20) and (block[0] <= snakehead[0])):
                    if(snakehead[1] == block[1]):
                        checksnakeleft = 1

                if ((snakehead[1] >= block[1] - 20) and (block[1] >= snakehead[1])):
                    if(snakehead[0] == block[0]):
                        checksnakedown = 1

                if ((snakehead[1] <= block[1] + 20) and (block[1] <= snakehead[1])):
                    if(snakehead[0] == block[0]):
                        checksnakeup = 1

            elif movingdown == 1:
                #danger down,left,right
                if ((snakehead[0] <= block[0] + 20) and (block[0] <= snakehead[0])):
                    if(snakehead[1] == block[1]):
                        checksnakeleft = 1

                if ((snakehead[0] >= block[0] - 20) and (block[0] >= snakehead[0])):
                    if(snakehead[1] == block[1]):
                        checksnakeright = 1

                if ((snakehead[1] >= block[1] - 20) and (block[1] >= snakehead[1])):
                    if(snakehead[0] == block[0]):
                        checksnakedown = 1

            elif movingup == 1:
                #danger up,left,right
                if ((snakehead[0] <= block[0] + 20) and (block[0] <= snakehead[0])):
                    if(snakehead[1] == block[1]):
                        checksnakeleft = 1

                if ((snakehead[0] >= block[0] - 20) and (block[0] >= snakehead[0])):
                    if(snakehead[1] == block[1]):
                       checksnakeright = 1

                if ((snakehead[1] <= block[1] + 20) and (block[1] <= snakehead[1])):
                    if(snakehead[0] == block[0]):
                       checksnakeup = 1


        # knowing that top left has least number 
        # and also top right has most number

        foodleft = 0
        foodright = 0
        foodup = 0 
        fooddown = 0

        if game.snake_position[0] > game.fruit_position[0]: 
            foodleft = 1
        if game.snake_position[0] < game.fruit_position[0]:
            foodright = 1
        if game.snake_position[1] > game.fruit_position[1]:
            foodup = 1
        if game.snake_position[1] < game.fruit_position[1]:
            fooddown = 1


        #setting everything in an array
        state = [
            movingleft,
            movingright,
            movingup,
            movingdown,
            foodleft,
            foodright,
            foodup,
            fooddown,
            CheckUp,
            CheckDown,
            CheckLeft,
            CheckRight,
            checksnakeleft,
            checksnakeright,
            checksnakeup,
            checksnakedown,
            ]


        # print(state)

        return np.array(state)


def Train():
    Brain = SnakeBrain()
    game = GameEnviroment()

    plotscores = []

    topscore = 0

    while True:


        #gets initial state
        initial_state = Brain.get_State(game)
        #print(initial_state)

        #gets the nextmove of the snake
        nextmove = Brain.snake_Action(initial_state)

        #gets the reward,check if dead, and score
        # basically makes the move in the enviroment
        reward,dead,score = game.play(nextmove)

        #gets the next state
        next_state = Brain.get_State(game)

        #training while playing
        Brain.train_while_play(initial_state,nextmove,reward,next_state,dead)

        #gathers and remembers
        Brain.remember(initial_state,nextmove,reward,next_state,dead)


        #if its dead it will reset the game, add up totalgames, start the long training
        #then if topscore save the best model
        #if after 1k games saves the performance and data
        if dead:
            game.reset()
            Brain.totalgames = Brain.totalgames + 1
            Brain.train_after_play()



            #check for topscore and possibly save this working model?
            if score > topscore:
                topscore = score
                #torch.save(Brain.model,'best_model.pth')

            print('Game',Brain.totalgames,'Score',score,'Record:',topscore)



            plotscores.append(score)
            plt.plot(plotscores)
           # if len(plotscores) == 1000:
               # np.savetxt('Run1Game1000.csv',plotscores,delimiter=',')






#starts the entire process
Train()






