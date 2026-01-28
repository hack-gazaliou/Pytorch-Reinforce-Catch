import pygame
import Engine
visualize = True

class View:
    
    
    def __init__(self, eng :Engine):
        self.engine = eng
        self.width = 640
        self.height = 700
        self.colors = {'font':(30,30,30), 'apple': (0,255,0), 'mango':(255,0,0), 'bomb':(255,255,0), 'paddle':(200,200,255), 'text':(255,255,255)}
        self.clock = pygame.time.Clock()
        self.running = True
        self.fruit_map = {0 : 'apple', 1 : 'mango', -1 : 'bomb'}
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        
    def render (self, eng :Engine) :
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        self.screen.fill(self.colors['font'])
        
        rect_w = eng.paddle_width * self.width
        rect_h = eng.paddle_height * self.height
        rect_x = (eng.paddle_x * self.width) - (rect_w / 2)
        rect_y = self.height - rect_h - 10 # 10px de marge en bas
        
        pygame.draw.rect(self.screen, self.colors['paddle'], (rect_x, rect_y, rect_w, rect_h))
        if len(eng.fruit_x) > 0:   
            for x, y, t in zip(eng.fruit_x, eng.fruit_y, eng.fruit_type) :
                px = x * self.width
                py = (1.0-y) * self.height -10
                rad = int(0.04 * self.width)
                color_key = self.fruit_map.get(t, 'apple')
                pygame.draw.circle(self.screen, self.colors[color_key], (px, py), rad)
                
        self._draw_ui(eng)
        pygame.display.flip()
        self.clock.tick(60)
        return True
        
        
def draw_ui(self, eng: Engine):
    info_text = f"Step: {eng.current_step}"
    info_surf = self.font.render(info_text, True, self.colors['text'])
    self.screen.blit(info_surf, (10, 10))  # en haut à gauche

    heart_text = self.font.render("♥", True, (255, 0, 0))  
    margin = 5
    for i in range(eng.lives):
        x = self.screen.get_width() - (i + 1) * (heart_text.get_width() + margin)
        y = 10  
        self.screen.blit(heart_text, (x, y))
 