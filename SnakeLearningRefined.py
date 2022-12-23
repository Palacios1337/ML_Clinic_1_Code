# importing libraries
import random
import pygame
import time
import random 
import win32com.client as comclt
import numpy as np



wsh= comclt.Dispatch("WScript.Shell")
#--------------------------------------------------------------------------------
snake_speed = 1000
 
# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)
 
# Initialising pygame
pygame.init()

# FPS (frames per second) controller
 
# fruit position

 
class GameEnviroment:


    def __init__(self,window_x = 200,window_y = 200 ):
        self.window_x = window_x
        self.window_y = window_y
        pygame.display.set_caption('Snake :I')
        self.game_window = pygame.display.set_mode((self.window_x,window_y))
        self.fps = pygame.time.Clock()
        self.reset()

    def _place_food(self):
        self.fruit_position = [random.randrange(1, (self.window_x//10)) * 10,
                          random.randrange(1, (self.window_y//10)) * 10]
        self.food = self.fruit_position
        self.fruit_spawn = True
        if self.food in self.snake_body:
            self._place_food()

    def show_score(self,choice, color, font, size):
   
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        self.game_window.blit(score_surface, score_rect)
 
# game over function
    def reset(self):
        self.score = 0
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.snake_position = [120, 120]
        self.snake_body = [[120, 120],
                           [110, 120],
                           [100, 120],
                           [90, 120]]
        self._place_food()
        while (len(self.snake_body) > 4):
            self.snake_body.pop()


    def snakymove(self,nextmove):

        if nextmove[0] == 1:
            self.change_to = 'LEFT'
        elif nextmove[1] == 1:
            self.change_to = 'RIGHT'
        elif nextmove[2] == 1:
            self.change_to = 'UP'
        elif nextmove[3] == 1:
            self.change_to = 'DOWN'

        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'    

    def makemovement(self):
        if self.direction == 'UP':
            self.snake_position[1] -= 10
        if self.direction == 'DOWN':
            self.snake_position[1] += 10
        if self.direction == 'LEFT':
            self.snake_position[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_position[0] += 10

    def checktouchbody(self):

       # Touching the snake body
       for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                return True


    def play(self,nextmove):
       Dead = False
       Reward = -.075

       self.snakymove(nextmove)
       self.makemovement()

       self.snake_body.insert(0, list(self.snake_position))
       if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
           Reward = 5
           self.score += 10
           self.fruit_spawn = False
       else:
           self.snake_body.pop()
         
       if not self.fruit_spawn:
           checkfruitposition = [random.randrange(1, (self.window_x//10)) * 10,random.randrange(1, (self.window_y//10)) * 10] 
           while checkfruitposition in self.snake_body:
               checkfruitposition = [random.randrange(1, (self.window_x//10)) * 10,random.randrange(1, (self.window_y//10)) * 10] 

           self.fruit_position = checkfruitposition
         
       self.fruit_spawn = True
       self.game_window.fill(black)
     
       for pos in self.snake_body:
            pygame.draw.rect(self.game_window, green,
                             pygame.Rect(pos[0], pos[1], 10, 10))
       pygame.draw.rect(self.game_window, white, pygame.Rect(
           self.fruit_position[0], self.fruit_position[1], 10, 10))
 
        # Game Over conditions
       if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            Reward = -15
            Dead = True
            return(Reward,Dead,self.score)

       if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            Reward = -15
            Dead = True
            return(Reward,Dead,self.score)
 
        #touch sanke body

       Dead = self.checktouchbody()
       if Dead:
           Reward = -20
           return(Reward,Dead,self.score)


        # displaying score countinuously
       self.show_score(1, white, 'times new roman', 20)
 
        # Refresh game screen
       pygame.display.update()
 
        # Frame Per Second /Refresh Rate
       self.fps.tick(snake_speed)

       return(Reward,Dead,self.score)

