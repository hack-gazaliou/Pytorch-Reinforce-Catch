import numpy as np
import random as rd

WIDTH = 1.0
HEIGHT = 1.0
PADDLE_WIDTH = 0.20
PADDLE_HEIGHT = 0.03

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
        self.lvl = 1
        
        self.fruit_type = [] #1 = mango, 0 = apple, -1 = bomb
        self.fruit_speed = 0.003
        self.paddle_speed = 0.01
        self.speed_multipliers = [-2, -1, 0 , 1, 2]
        self.last_fall = 0
        
        self.current_step  = 0
        self.score = 0
        self.max_steps = 500
        self.done = False
  
    def get_observation(self) :
        sorted_indices = np.argsort(self.fruit_y)
        nb_fruits = len(self.fruit_y)
        if nb_fruits > 0:
            idx1 = sorted_indices[0] 
            obs_f1 = [
                self.fruit_x[idx1],     #x
                self.fruit_y[idx1],     #y
                self.fruit_type[idx1],  #type
                1.0                     #presence flag
            ]
        else : obs_f1 = [0.0, 0.0, 0.0, 0.0] 
        
        if nb_fruits > 1:
            idx2 = sorted_indices[1] 
            obs_f2 = [
                self.fruit_x[idx2], 
                self.fruit_y[idx2], 
                self.fruit_type[idx2], 
                1.0 
            ]
        else:
            obs_f2 = [0.0, 0.0, 0.0, 0.0]
                    
        if nb_fruits > 2:
            idx3 = sorted_indices[2] 
            obs_f3 = [
                self.fruit_x[idx3], 
                self.fruit_y[idx3], 
                self.fruit_type[idx3], 
                1.0
            ]
        else:
            obs_f3 = [0.0, 0.0, 0.0, 0.0]   
            
        if nb_fruits > 3:
            idx4 = sorted_indices[3] 
            obs_f4 = [
                self.fruit_x[idx4], 
                self.fruit_y[idx4], 
                self.fruit_type[idx4], 
                1.0 
            ]
        else:
            obs_f4 = [0.0, 0.0, 0.0, 0.0]
        
        return np.array([
        self.paddle_x,
        *obs_f1,
        *obs_f2,
        *obs_f3,
        *obs_f4,
        self.lives / self.max_lives 
        ])

    def type_prob(self):
        if self.lvl == 1:
            return 0.2 , 0.6, 0.2 #p_bomb, p_apple, p_mango
        elif self.lvl == 2:
            return 0.25, 0.50, 0.25
        elif self.lvl == 3:
            return 0.3, 0.4, 0.3
                    
    def spawn_fruit(self):
        self.fruit_x.append(rd.random())
        self.fruit_y.append(1.0)
        
        p_bomb, p_apple, p_mango = self.type_prob()
        r = rd.random()

        if r < p_bomb:
            self.fruit_type.append(-1)
        elif r < p_bomb + p_apple:
            self.fruit_type.append(0)
        else:
            self.fruit_type.append(1)

        self.last_fall = self.current_step
        
    def spawn_interval(self):
        match self.lvl :
            case 1 :
                return 95
            case 2 :
                return 65
            case 3 :
                return 35
    
    def reset(self): 
        self.lives = self.max_lives
        self.paddle_x = 0.5
        self.fruit_x = []
        self.fruit_y = []
        self.fruit_type = []
        self.current_step = 0
        self.score = 0
        self.lvl = rd.randint(1,3)
        return self.get_observation() #return the initial observation

           
    def step(self, action):
        reward = 0
        self.done = False
        if self.paddle_x + self.speed_multipliers[action]*self.paddle_speed >1 : 
            self.paddle_x = 1
        elif self.paddle_x + self.speed_multipliers[action]*self.paddle_speed <0 : 
            self.paddle_x = 0
        else : 
            self.paddle_x += self.speed_multipliers[action]*self.paddle_speed
        self.fruit_y = [y-self.fruit_speed for y in self.fruit_y]
        if len(self.fruit_y) > 0:
            in_paddle = self.paddle_x - self.paddle_width/2 <= self.fruit_x[0] <= self.paddle_x + self.paddle_width/2
            if self.fruit_y[0] <= 0.045 and in_paddle:
                    match self.fruit_type[0]:
                        case 1: #mango
                            reward = 2
                            self.score +=2
                        case 0: #apple
                            reward = 1
                            self.score+=1
                        case -1: #bomb
                            reward = -3
                            self.lives = 0
                            self.done = True
                    self.fruit_x.pop(0)
                    self.fruit_y.pop(0)
                    self.fruit_type.pop(0)

            elif self.fruit_y[0] <= 0:   
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
                        self.done = True
        self.current_step += 1
        if self.current_step >= self.max_steps and self.done == False:
            self.done = True
        if not self.done :
            interval = self.spawn_interval()
            jitter = rd.randint(-5, 5)
            if self.current_step - self.last_fall >= interval + jitter or self.current_step == 1: #to have a fruit at the beginning
                self.spawn_fruit()
        return self.get_observation(), reward, self.done