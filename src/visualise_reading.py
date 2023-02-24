__author__ = 'Sam van Leipsig'

import pygame
import sys
from reading_functions import calc_acuity,get_attention_skewed

# Define some colors
BLACK = (   0,   0,   0)
WHITE = ( 255, 255, 255)
GREEN = (   0, 255,   0)
RED = ( 255,   0,   0)
YELLOW = (255,255,0)
GREY = (220,220,220)

# Set the height and width of the screen
size = [800, 200]
width = 20
height = 20
margin = 1
eyeposition = 15
span_size = 10.


pygame.init()
# initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
myfont = pygame.font.Font(None, 32)
screen = pygame.display.set_mode(size)
pygame.display.toggle_fullscreen()
pygame.display.set_caption("Array Backed Grid")
clock = pygame.time.Clock()

def calc_contrast(monogram_position, eye_position, attention_position, attend_width, attention_skew, let_per_deg):

    attention_eccentricity = monogram_position-attention_position
    eye_eccentricity = abs(monogram_position-eye_position)
    attention = get_attention_skewed(attend_width, attention_eccentricity, attention_skew)
    visual_acuity = calc_acuity(eye_eccentricity, let_per_deg)

    return attention * visual_acuity

class Grid(object):
    def __init__(self,screensize,width,height,margin):
        self.screen  = screensize
        self.width = width
        self.height = height
        self.margin = margin
        self.number_rows = 10
        self.number_colums = screensize[0]/ (width+margin)
        self.grid = []
    def init_gridarray(self):
        for row in range(self.number_rows):
            self.grid.append([])
            for column in range(self.number_colums):
                self.grid[row].append(0)
    def draw(self):
        for row in range(self.number_rows):
            for column in range(self.number_colums):
                pygame.draw.rect(screen,WHITE,
                             [(self.margin+self.width)*column+self.margin,
                              (self.margin+self.height)*row+self.margin,
                              self.width,self.height])


class Stimulus(object):
    def __init__(self):
        self.eyeposition = 5.
        self.stimulus = ' Beginning to read'
        self.attentional_span = 10.
        self.attentionposition = self.eyeposition
        self.fixation = 0
        self.attention_skew = 0
        self.let_per_deg = 0
    def update_stimulus(self,stimulus,eyepos,att_span,attpos,fixation,attention_skew,let_per_deg):
        self.stimulus = stimulus
        self.eyeposition = eyepos
        self.attentional_span = att_span
        self.attentionposition = attpos
        self.fixation = fixation+1
        self.attention_skew = attention_skew
        self.let_per_deg = let_per_deg

    def draw(self):
        if self.eyeposition != self.attentionposition:
            pygame.draw.rect(screen,GREEN,
                     [(margin+width)*(self.attentionposition-1)+margin,
                      (margin+height)*4+margin, width,height])
        for pos,value in enumerate(self.stimulus):
            letter = myfont.render(value, 1, BLACK, WHITE)
            if self.eyeposition - pos == 0:
                pygame.draw.rect(screen,YELLOW,
                         [((margin+width) * (pos-1)) + margin,
                          ((margin+height) * 4) + margin,
                          width, height])
            contrast_change = calc_contrast(pos,self.eyeposition,self.attentionposition,self.attentional_span,self.attention_skew,self.let_per_deg)
            letter.set_alpha(450 * contrast_change)
            letterrect = letter.get_rect()
            letterrect.centerx = (margin+width)* pos - (width/2)
            letterrect.centery = (margin+height)* 5 - (height/2)
            screen.blit(letter, letterrect)
    def draw_span(self):
        x = (margin+width)*(self.eyeposition-(self.attentional_span/2))
        y = (margin+(height))* 4 - (height/2)
        spansize = (margin+width)*self.attentional_span
        pygame.draw.ellipse(screen, BLACK, [x, y, spansize, 40], 1)
    def draw_fixation_number(self):
        number = myfont.render(str(self.fixation), 1, BLACK, WHITE)
        numberrect = number.get_rect()
        numberrect.centerx = (margin+width)* self.eyeposition - (width/2)
        numberrect.centery = (margin+height)* 3 - (height/2)
        screen.blit(number, numberrect)
    def draw_arrow(self):
        number = myfont.render(str("|"), 1, BLACK, WHITE)
        numberrect = number.get_rect()
        numberrect.centerx = (margin+width)* self.eyeposition - (width/2)
        numberrect.centery = (margin+height)* 3 - (height/2)
        screen.blit(number, numberrect)


## INIT
grid = Grid(size,width,height,margin)
grid.init_gridarray()
stimulus = Stimulus()

def update_stimulus(newstimulus,eyeposition,attentional_span,attentionposition,fixation):
    stimulus.update_stimulus(newstimulus,float(eyeposition),float(attentional_span),float(attentionposition),fixation)

def save_screen(fixationcounter,shift):
    pygame.image.save(screen, "Screenshots/Screen"+ str(fixationcounter) + str(shift) + ".jpg")

# -------- Main Program Loop -----------
def main():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            sys.exit()

    # Set the screen background
    screen.fill(GREY)
    grid.draw()
    stimulus.draw()
    #stimulus.draw_span()
    #stimulus.draw_fixation_number()
    stimulus.draw_arrow()

    # update the screen
    clock.tick(60)
    pygame.display.update()
    #pygame.display.flip()

if __name__ == '__main__nc': main()
