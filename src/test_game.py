import numpy as np
import random as rd
import pygame
from engine import engine
import sys

WIDTH = 1.0
HEIGHT = 1.0
PADDLE_WIDTH = 0.20
PADDLE_HEIGHT = 0.03
MINIMAL_DELAY_1 = 80
MINIMAL_DELAY_2 = 50
MINIMAL_DELAY_3 = 25

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
        self.max_steps = 10000
        self.exploding = False
        self.explosion_timer = 0
        self.explosion_duration = 30  # frames (~0.5s à 60 FPS)
        self.explosion_x = None
        self.explosion_y = None
        self.sp = 0
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))


        #self.explosion_frames = [pygame.image.load("explosion.png").convert_alpha()for i in range(1, 6)]
        #self.explosion_frames = [pygame.transform.scale(img, (80, 80))for img in self.explosion_frames]
        
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
        if self.current_step < 1000:
            return 0 , 0.85, 0.15 #p_bomb, p_apple, p_mango
        elif self.current_step < 4000:
            return 0.15, 0.7, 0.15
        elif self.current_step < 6000:
            return 0.2, 0.65, 0.15
        elif self.current_step < 8000:
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
        if self.current_step < 1000:
            return 280
        elif self.current_step < 2000:
            return 230
        elif self.current_step < 3000:
            return 150
        elif self.current_step <4000:
            return 100 
        elif self.current_step <5000:
            return 80        
        elif self.current_step <6000:
            return 70
        elif self.current_step <8000:
            return 60
        elif self.current_step <9000:
            return 40
        else:
            return 30    # enfer
    
    def reset(self): #reset the game to start a new episode
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
        """if self.exploding:
            self.explosion_timer += 1
            if self.explosion_timer >= self.explosion_duration:
                self.lives = 0
                return self.get_observation(), -3, True
            return self.get_observation(), 0, False"""""
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
                
            """spawn_probability = max(min(0.0005* (self.current_step**0.333), 0.032), 0.005)
            self.sp = 1/(60*spawn_probability)
            #spawn_probability = min(0.0082 * self.current_step**0.1, 0.15)
            delay = 0
            if self.current_step > 6000 :
                delay = MINIMAL_DELAY_3
            elif self.current_step > 3000 :
                delay = MINIMAL_DELAY_2 
            else : delay = MINIMAL_DELAY_1"""
        return self.get_observation(), reward, done

visualize = True

class View:
        
    def __init__(self, eng :engine):
        self.engine = eng
        self.width = 640
        self.height = 700
        self.colors = {'font':(200,200,255), 'apple': (0,255,0), 'mango':(255,0,0), 'bomb':(255,255,0), 'paddle':(91,60,17), 'text':(0,0,0)}
        self.clock = pygame.time.Clock()
        self.running = True
        self.fruit_map = {0 : 'apple', 1 : 'mango', -1 : 'bomb'}
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.bomb_img = pygame.image.load("bomb2.png").convert_alpha()
        self.bomb_img = pygame.transform.scale(self.bomb_img,(int(0.09 * self.width), int(0.09 * self.width)))
        self.mango_img = pygame.image.load("mango.png").convert_alpha()
        self.mango_img = pygame.transform.scale(self.mango_img,(int(0.12 * self.width), int(0.12 * self.width)))
        self.apple_img = pygame.image.load("apple.png").convert_alpha()
        self.apple_img = pygame.transform.scale(self.apple_img,(int(0.08 * self.width), int(0.08 * self.width)))
        self.cloud_img = pygame.image.load("cloud.png").convert_alpha()
        self.coud_img = pygame.transform.scale(self.cloud_img,(int( 0.3*self.width), int(0.3*self.width)))

        
    def render (self, eng :engine) :
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        self.screen.fill(self.colors['font'])
        
        rect_w = eng.paddle_width * self.width
        rect_h = eng.paddle_height * self.height
        rect_x = (eng.paddle_x * self.width) - (rect_w / 2)
        rect_y = self.height - rect_h - 10 # 10px de marge en bas
        nuage1 = self.cloud_img.get_rect(center=(450,75 ))
        self.screen.blit(self.cloud_img, nuage1) 
        
        pygame.draw.rect(self.screen, self.colors['paddle'], (rect_x, rect_y, rect_w, rect_h))
        if len(eng.fruit_x) > 0:   
            for x, y, t in zip(eng.fruit_x, eng.fruit_y, eng.fruit_type) :
                px = x * self.width
                py = (1.0-y) * self.height -10
                rad = int(0.04 * self.width)
                color_key = self.fruit_map.get(t, 'apple')
                if t == -1:  # bombe
                    rect = self.bomb_img.get_rect(center=(px, py))
                    self.screen.blit(self.bomb_img, rect)
                elif t == 0 : #apple
                    rect = self.apple_img.get_rect(center=(px, py))
                    self.screen.blit(self.apple_img, rect)                    
                else:
                    rect = self.mango_img.get_rect(center=(px, py))
                    self.screen.blit(self.mango_img, rect)                      
                    """rad = int(0.04 * self.width)
                    color_key = self.fruit_map.get(t, 'apple')
                    pygame.draw.circle(self.screen, self.colors[color_key], (px, py), rad)"""
                            
        self._draw_ui(eng)
        pygame.display.flip()
        self.clock.tick(60)
        return True
        
    def _draw_ui(self, eng: engine):
        info_text = f"Step: {eng.current_step}"
        info_score = f"Score: {eng.score}"
        info_timing = f"Δt: {round(eng.spawn_interval()/60, 2)} s"

        p_bomb, p_apple, p_mango = eng.type_prob()
        info_proba = (
            f"ap:{int(p_apple*100)}% "
            f"ma:{int(p_mango*100)}% "
            f"bo:{int(p_bomb*100)}%"
        )

        info_surf = self.font.render(info_text, True, self.colors['text'])
        info_surf_2 = self.font.render(info_score, True, self.colors['text'])
        info_surf_3 = self.font.render(info_timing, True, self.colors['text'])
        info_surf_4 = self.font.render(info_proba, True, self.colors['text'])

        self.screen.blit(info_surf, (10, 10))
        self.screen.blit(info_surf_2, (10, 40))
        self.screen.blit(info_surf_3, (10, 70))
        self.screen.blit(info_surf_4, (10, 100))

        heart_text = self.font.render("♥", True, (255, 0, 0))  
        margin = 5
        for i in range(eng.lives):
            x = self.screen.get_width() - (i + 1) * (heart_text.get_width() + margin)
            y = 10  
            self.screen.blit(heart_text, (x, y))
    
    


if __name__ == "__main__":
    env = engine()
    view = View(env)
    
    state = env.reset()
    running = True
    
    print("=== DÉBUT DU TEST ===")
    print("Utilise les Flèches GAUCHE / DROITE pour bouger.")
    
    while running:
        if env.current_step == 50:
            env.spawn_fruit()
        
        # 1. Lire le clavier (Human Agent)
        keys = pygame.key.get_pressed()
        
        # Mapping Clavier -> Action Index
        # Actions: 0:Left++, 1:Left, 2:Stop, 3:Right, 4:Right++
        if keys[pygame.K_LEFT]:
            action = 1 # Gauche Rapide
        elif keys[pygame.K_RIGHT]:
            action = 3 # Droite Rapide
        else:
            action = 2 # Stop
            
        # 2. Update Physique
        obs, reward, done = env.step(action)
        # 3. Update Visuel
        keep_open = view.render(env)
        if not keep_open or done:
            running = False
            if done: print("GAME OVER")

    pygame.quit()
    sys.exit()