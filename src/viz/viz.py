from decimal import Decimal

import pygame
from pygame.locals import *
from math import *
import numpy as np
import argparse

COLOR = {'black': (0, 0, 0),
         'red': (255, 0, 0),
         'green': (0, 255, 0),
         'blue': (0, 0, 255),
         'white': (255, 255, 255)}

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
SCREEN_DIM = (SCREEN_WIDTH, SCREEN_HEIGHT)
SCREEN_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)


class Pendulum(pygame.sprite.Sprite):
    def __init__(self, pivot_vect, length, bob_radius, data):
        pygame.sprite.Sprite.__init__(self)

        self.font = pygame.font.SysFont("Ubuntu", 24)

        self.length = length
        self.bob_radius = bob_radius
        self.index = 0
        self.data = data

        swinglen = length + bob_radius

        self.image = pygame.Surface((swinglen * 2, swinglen * 2)).convert()
        self.rect = self.image.get_rect()
        self.rect.topleft = (pivot_vect[0] - swinglen, pivot_vect[1] - swinglen)  # place so that pivot is at center

        self.bob_x = int(length * sin(self._angle()) + self.rect.width // 2)
        self.bob_y = int(length * cos(self._angle()) + self.rect.height // 2)
        self.bob_rect = None

        self._render()

    def _render(self):
        self.image.fill(COLOR['black'])
        bob_pos = (self.bob_x, self.bob_y)

        pygame.draw.aaline(self.image, COLOR['red'], (self.rect.width // 2, self.rect.height // 2), bob_pos, True)

        self.bob_rect = pygame.draw.circle(self.image, COLOR['blue'], bob_pos, self.bob_radius, 0)
        x1, y1 = self.bob_rect.topleft
        x2, y2 = self.rect.topleft

        self.bob_rect.topleft = (x1 + x2, y1 + y2)

        text = self.font.render(f"{self._normalize(self._angle() + pi):.4E}", True, COLOR['white'])
        self.image.blit(text, (0, 0))

    @staticmethod
    def _normalize(angle):
        return ((angle + pi) % (2 * pi)) - pi

    def _angle(self):
        return self._normalize(self.data[self.index][1] + pi)

    def update(self):
        # coords relative to pivot
        self.index = (self.index + 1) % len(self.data)
        angle = self._angle()
        length = self.length

        x = int(length * sin(angle))
        y = int(length * cos(angle))
        self.bob_x = x + self.rect.width // 2
        self.bob_y = y + self.rect.height // 2
        self._render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="location of csv containing simulation data")
    args = parser.parse_args()

    if not args.filename:
        return

    data = np.genfromtxt(args.filename, delimiter=',', skip_header=1)

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("Pendulum Simulation")

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill(COLOR['black'])
    clock = pygame.time.Clock()

    pendulum = Pendulum(pivot_vect=SCREEN_CENTER, length=300, bob_radius=25, data=data)
    group = pygame.sprite.RenderPlain((pendulum,))

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and (event.key == K_ESCAPE or event.key == K_q):
                return
        group.update()
        screen.blit(background, (0, 0))
        group.draw(screen)
        pygame.display.flip()


if __name__ == '__main__':
    main()
