import pygame
from engine import engine
import os

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
            
    def save_frame(self, epoch, frame_idx, folder="recordings"):
            path = f"{folder}/epoch_{epoch}"
            if not os.path.exists(path):
                os.makedirs(path)
            filename = f"{path}/frame_{frame_idx:04d}.png"
            pygame.image.save(self.screen, filename)
    
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