import numpy as np
import random as rd
import pygame
from engine import engine
import sys
import time
import os 
WIDTH = 1.0
HEIGHT = 1.0
PADDLE_WIDTH = 0.17
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
        
        self.max_lives = 3
        self.lives = 3
        
        self.paddle_x = 0.5
        self.fruit_x = []
        self.fruit_y = []
        
        self.fruit_type = [] #1 = mango, 0 = apple, -1 = bomb
        self.fruit_speed = 0.003
        self.paddle_speed = 0.01
        self.speed_multipliers = [-2, -1, 0 , 1, 2]
        self.last_fall = 0
        
        self.current_step  = 0
        self.score = 0
        self.max_steps = 10000
        self.lvl = 1
        self.done = False
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
            
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
    
    def reset(self): #reset the game to start a new episode
        self.lives = self.max_lives
        self.paddle_x = 0.5
        
        # On vide les listes
        self.fruit_x = []
        self.fruit_y = []
        self.fruit_type = []
        
        tiers_ranges = [(0.33, 0.50), (0.50, 0.66), (0.66, 0.9)]
        
        for y_min, y_max in tiers_ranges:
            self.fruit_x.append(rd.random())
            self.fruit_y.append(rd.uniform(y_min, y_max)) 
            p_bomb, p_apple, p_mango = self.type_prob()
            r = rd.random()
            if r < p_bomb:
                self.fruit_type.append(-1)
            elif r < p_bomb + p_apple:
                self.fruit_type.append(0)
            else:
                self.fruit_type.append(1)

        combined = list(zip(self.fruit_y, self.fruit_x, self.fruit_type))
        combined.sort(key=lambda x: x[0]) 
        
        self.fruit_y, self.fruit_x, self.fruit_type = zip(*combined)
        self.fruit_x = list(self.fruit_x)
        self.fruit_y = list(self.fruit_y)
        self.fruit_type = list(self.fruit_type)

        self.current_step = 0
        self.score = 0
        self.lvl = 2        
        return self.get_observation() 
           
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
            if self.current_step - self.last_fall >= interval + jitter:
                self.spawn_fruit()
        return self.get_observation(), reward, self.done

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
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        img_dir = os.path.join(root_dir, "pictures")
        self.bomb_img = pygame.image.load(os.path.join(img_dir, "bomb2.png")).convert_alpha()
        self.bomb_img = pygame.transform.scale(self.bomb_img, (int(0.09 * self.width), int(0.09 * self.width)))
        
        self.mango_img = pygame.image.load(os.path.join(img_dir, "mango.png")).convert_alpha()
        self.mango_img = pygame.transform.scale(self.mango_img, (int(0.12 * self.width), int(0.12 * self.width)))
        
        self.apple_img = pygame.image.load(os.path.join(img_dir, "apple.png")).convert_alpha()
        self.apple_img = pygame.transform.scale(self.apple_img, (int(0.08 * self.width), int(0.08 * self.width)))
        
        self.cloud_img = pygame.image.load(os.path.join(img_dir, "cloud.png")).convert_alpha()
        self.cloud_img = pygame.transform.scale(self.cloud_img, (int(0.3 * self.width), int(0.3 * self.width)))

        
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
                            
        self._draw_ui(eng)
        pygame.display.flip()
        self.clock.tick(60)
        return True
        
    def _draw_ui(self, eng: engine):
        info_text = f"Step: {eng.current_step}"
        info_score = f"Score: {eng.score}"
        info_timing = f"Δt: {round(eng.spawn_interval()/60, 2)} s"
        info_lvl = f"Lvl :{eng.lvl}"
        
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
        info_surf_5 = self.font.render(info_lvl, True, self.colors['text'])
        
        self.screen.blit(info_surf, (10, 10))
        self.screen.blit(info_surf_2, (10, 40))
        self.screen.blit(info_surf_3, (10, 70))
        self.screen.blit(info_surf_4, (10, 100))
        self.screen.blit(info_surf_5, (10, 130))

        heart_text = self.font.render("♥", True, (255, 0, 0))  
        margin = 5
        for i in range(eng.lives):
            x = self.screen.get_width() - (i + 1) * (heart_text.get_width() + margin)
            y = 10  
            self.screen.blit(heart_text, (x, y))
            
    def show_game_over(self, score):
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128) 
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        font_big = pygame.font.SysFont("Arial", 60, bold=True)
        text_go = font_big.render("GAME OVER", True, (255, 0, 0))
        rect_go = text_go.get_rect(center=(self.width // 2, self.height // 2 - 30))
        self.screen.blit(text_go, rect_go)
        
        text_score = self.font.render(f"Final Score: {score}", True, (255, 255, 255))
        rect_score = text_score.get_rect(center=(self.width // 2, self.height // 2 + 30))
        self.screen.blit(text_score, rect_score)
        pygame.display.flip()
    

if __name__ == "__main__":
    env = engine()
    view = View(env)
    
    state = env.reset()
    running = True
    
    print("=== DÉBUT DU TEST ===")
    print("Utilise les Flèches GAUCHE / DROITE pour bouger.")
    
    while running:
        
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
            if done: 
                print("GAME OVER")
                view.show_game_over(env.score)
                time.sleep(3)

    pygame.quit()
    sys.exit()