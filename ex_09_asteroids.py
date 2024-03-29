from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
import pygame
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from vector_classes import (Vector, Vec1, Vec2, Vec3, Vec6, CarForSale, Matrix5_by_3, 
    Matrix, ImageVector, Function, Function2, LinearFunction, QuadraticFunction, Polynomial,
    PolygonModel, Ship, Asteroid)
from mfp_modules import colors, vector_drawing, vectors
import sys

# HELPERS / SETTINGS

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE =  (0, 0, 255)
GREEN = (0, 255, 0)
RED =   (255, 0, 0)
LIGHT_GRAY =  (240, 240, 240)
DARK_GRAY = (128, 128, 128)
width, height = 800, 800
asteroid_count = 20
acceleration = 3
screenshot_mode = True

default_asteroids = [Asteroid() for _ in range(0,asteroid_count)]
for ast in default_asteroids:
    ast.x = randint(-9,9)
    ast.y = randint(-9,9)
ship = Ship()
laser = ship.laser_segment()

###written
def to_pixels(x,y):
    return (width/2 + width * x / 20, height/2 - height * y / 20)

###written
def draw_poly(screen, polygon_model, color=BLACK):
    pixel_points = [to_pixels(x,y) for x,y in polygon_model.transformed()]
    pygame.draw.lines(screen, color, True, pixel_points, 2)
    if polygon_model.draw_center:
        cx, cy = to_pixels(polygon_model.x, polygon_model.y)
        pygame.draw.circle(screen, BLACK, (int(cx), int(cy)), 4, 4)

def draw_segment(screen, v1,v2,color=RED):
    pygame.draw.line(screen, color, to_pixels(*v1), to_pixels(*v2), 2)

def draw_grid(screen):
    for x in range(-9,10):
        draw_segment(screen, (x,-10), (x,10), color=LIGHT_GRAY)
    for y in range(-9,10):
        draw_segment(screen, (-10, y), (10, y), color=LIGHT_GRAY)

    draw_segment(screen, (-10, 0), (10, 0), color=DARK_GRAY)
    draw_segment(screen, (0, -10), (0, 10), color=DARK_GRAY)



def main(asteroids=default_asteroids):

    pygame.init()

    screen = pygame.display.set_mode([width,height])

    pygame.display.set_caption("Asteroids!")

    done = False
    clock = pygame.time.Clock()

    # p key prints screenshot (you can ignore this variable)
    p_pressed = False

    while not done:

        clock.tick()

        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done=True # Flag that we are done so we exit this loop

        # UPDATE THE GAME STATE

        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()

        for ast in asteroids:
            ast.move(milliseconds)

        if keys[pygame.K_LEFT]:
            ship.rotation_angle += milliseconds * (2*pi / 1000)

        if keys[pygame.K_RIGHT]:
            ship.rotation_angle -= milliseconds * (2*pi / 1000)

        if keys[pygame.K_UP]:
            ax = acceleration * cos(ship.rotation_angle)
            ay = acceleration * sin(ship.rotation_angle)
            ship.vx += ax * milliseconds/1000
            ship.vy += ay * milliseconds/1000

        elif keys[pygame.K_DOWN]:
            ax = - acceleration * cos(ship.rotation_angle)
            ay = - acceleration * sin(ship.rotation_angle)
            ship.vx += ax * milliseconds/1000
            ship.vy += ay * milliseconds/1000


        # p key saves screenshot (you can ignore this)
        if keys[pygame.K_p] and screenshot_mode:
            p_pressed = True
        elif p_pressed:
            pygame.image.save(screen, 'figures/asteroid_screenshot_%d.png' % milliseconds)
            p_pressed = False


        ship.move(milliseconds)


        laser = ship.laser_segment()

        # DRAW THE SCENE

        screen.fill(WHITE)

        draw_grid(screen)

        if keys[pygame.K_SPACE]:
            draw_segment(screen, *laser)

        draw_poly(screen,ship)

        for asteroid in asteroids:
            if keys[pygame.K_SPACE] and asteroid.does_intersect(laser):
                asteroids.remove(asteroid)
            else:
                draw_poly(screen, asteroid, color=GREEN)


        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    if '--screenshot' in sys.argv:
        screenshot_mode = True
    main()
