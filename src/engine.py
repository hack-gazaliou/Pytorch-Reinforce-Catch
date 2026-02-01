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
        
        self.max_lives = 4
        self.lives = 3
        
        self.paddle_x = 0.5
        self.fruit_x = []
        self.fruit_y = []
        self.can_fall = True
        
        self.fruit_type = [] #1 = mango, 0 = apple, -1 = bomb
        self.fruit_speed = 0.003
        self.paddle_speed = 0.01
        self.speed_multipliers = [-2, -1, 0 , 1, 2]
        self.last_fall = 0
        
        self.current_step  = 0
        self.score = 0
        self.max_steps = 5000
        self.exploding = False
        self.sp = 0

        
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
    def type_prob(self):
        if self.current_step < 1000/2:
            return 0 , 0.85, 0.15 #p_bomb, p_apple, p_mango
        elif self.current_step < 4000/2:
            return 0.15, 0.7, 0.15
        elif self.current_step < 6000/2:
            return 0.2, 0.65, 0.15
        elif self.current_step < 8000/2:
            return 0.25, 0.55, 0.2
        else:
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
        if self.current_step < 1000/2:
            return 230
        elif self.current_step < 2000/2:
            return 190
        elif self.current_step < 3000/2:
            return 150
        elif self.current_step <4000/2:
            return 100 
        elif self.current_step <5000/2:
            return 80        
        elif self.current_step <6000/2:
            return 70
        elif self.current_step <8000/2:
            return 60
        elif self.current_step <9000/2:
            return 40
        else:
            return 30 
    
    def reset(self): 
        self.lives = self.max_lives
        self.paddle_x = 0.5
        self.fruit_x = []
        self.fruit_y = []
        self.fruit_type = []
        self.current_step = 0
        self.score = 0
        self.sp = 0
        return self.get_observation() #return the initial observation

    def change_type(self):
        self.fruit_type[-1] = 0 #transformation of bombs into apple for the beginning of the game
           
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
                            self.exploding = True
                            self.explosion_timer = 0
                            self.explosion_x = self.fruit_x[0]
                            self.explosion_y = self.fruit_y[0]
                            self.lives = 0
                            done = True
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
                        done = True
        self.current_step += 1
        if self.current_step >= self.max_steps and done == False:
            done = True
        if not done :
            interval = self.spawn_interval()
            jitter = rd.randint(-5, 5)
            if self.current_step - self.last_fall >= interval + jitter:
                self.spawn_fruit()
        return self.get_observation(), reward, done