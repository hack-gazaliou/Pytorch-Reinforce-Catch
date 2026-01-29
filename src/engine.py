import numpy as np
import random as rd

WIDTH = 1.0
HEIGHT = 1.0
PADDLE_WIDTH = 0.2
PADDLE_HEIGHT = 0.05

class engine:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.paddle_width = PADDLE_WIDTH
        self.paddle_height = PADDLE_HEIGHT
        
        self.max_lives = 3
        self.lives = 3
        
        self.paddle_x = 0.5
        self.fruit_x = []
        self.fruit_y = []
        
        self.fruit_type = [] #1 = mango, 0 = apple, -1 = bomb
        self.fruit_speed = 0.03
        self.paddle_speed = 0.05
        self.speed_multipliers = [-2, -1, 0 , 1, 2]
        
        self.current_step  = 0
        self.max_steps = 1000

        
    def get_observation(self):
        sorted_indices = np.argsort(self.fruit_y)
        nb_fruits = len(self.fruit_y)
        if nb_fruits > 0:
            idx1 = sorted_indices[0] 
            obs_f1 = [
                self.fruit_x[idx1], 
                self.fruit_y[idx1], 
                self.fruit_type[idx1]
            ]
        else: obs_f1 = [0.5, 1.5, 0] # No fruit in the frame, we display an out of screen virtual fruit
        if nb_fruits > 1:
            idx2 = sorted_indices[1] 
            obs_f2 = [
                self.fruit_x[idx2], 
                self.fruit_y[idx2], 
                self.fruit_type[idx2], 
                1.0 #flag second fruit
            ]
        else:
            obs_f2 = [0.0, 0.0, 0.0, 0.0] # <--- FLAG : "no second fruit"
            
        return np.array([
        self.paddle_x,
        *obs_f1,
        *obs_f2,
        self.lives / self.max_lives
    ])
            
    def spawn_fruit(self):
        self.fruit_x.append(rd.random())
        self.fruit_y.append(1.0)
        r = rd.random()
        if r > 0.8: self.fruit_type.append(-1) #bomb (20%)
        elif r > 0.2: self.fruit_type.append(0)#apple (60%)
        else: self.fruit_type.append(1) #mango (20%)
    
    def reset(self): #reset the game to start a new episode
        self.lives = self.max_lives
        self.paddle_x = 0.5
        self.fruit_x = [0.5]
        self.fruit_y = [1]
        self.fruit_type = [0]
        self.current_step = 0
        return self.get_observation() #return the initial observation
    
    def step(self, action):
        reward = 0
        done = False
        if self.paddle_x + self.speed_multipliers[action]*self.paddle_speed >1 : 
            self.paddle_x = 1
        elif self.paddle_x + self.speed_multipliers[action]*self.paddle_speed <0 : 
            self.paddle_x = 0
        else : 
            self.paddle_x += self.speed_multipliers[action]*self.paddle_speed
        self.fruit_y = [y-self.fruit_speed for y in self.fruit_y]
        if len(self.fruit_y) > 0:
            if self.fruit_y[0] <= 0:
                if self.paddle_x - self.paddle_width/2 <= self.fruit_x[0] <= self.paddle_x + self.paddle_width/2: #fruit is catched
                    match self.fruit_type[0]:
                        case 1: #mango
                            reward = 2
                        case 0: #apple
                            reward = 1
                        case -1: #bomb
                            reward = -3
                            self.lives = 0
                            done = True
                    self.fruit_x.pop(0)
                    self.fruit_y.pop(0)
                    self.fruit_type.pop(0)

                else:   #fruit falls
                    match self.fruit_type[0]:
                        case 1: #mango
                            reward = -2
                            self.lives -= 1
                        case 0: #apple
                            reward = -1
                            self.lives -= 1
                        case -1: #bomb
                            reward = 1
                    self.fruit_x.pop(0)
                    self.fruit_y.pop(0)
                    self.fruit_type.pop(0)
                    if self.lives <= 0:
                        done = True
        self.current_step += 1
        if self.current_step >= self.max_steps and done == False:
            done = True
        if not done :
            spawn_probability = min(0.03 + 0.00005 * self.current_step, 0.5)
            if rd.random() < spawn_probability:
                self.spawn_fruit()


        return self.get_observation(), reward, done

    