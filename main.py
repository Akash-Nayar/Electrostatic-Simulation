import pygame, sys, math
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances

K = 9 * (10 ** 3)

display_width = 1280
display_height = 720


particle_img = pygame.image.load('images/ball.png')


FPS = 30

gameDisplay = pygame.display.set_mode((display_width, display_height))

clock = pygame.time.Clock()


def to_int(tup):
    return tuple(map(int, tup))




# Define Vector class:
class Vector():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def coord(self):
        return [self.x, self.y]

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, num):
        return Vector(self.x * num, self.y * num)

    def __truediv__(self, num):
        return Vector(self.x / num, self.y / num)

    # Returns distance
    def __matmul__(self, other):
        return euclidean_distances([self.coord()], [other.coord()])[0]

    # Returns magnitude
    def __abs__(self):
        return euclidean_distances([self.coord()], [(0, 0)])[0]

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'Vector(x={self.x}, y={self.y})'


# Define PointCharge class
class PointCharge():

    def __init__(self, x, y, charge):
        self.pos = Vector(x, y)
        self.charge = charge
        self.color = (0, 0, 255) if charge > 0 else (255, 0, 0,)  # Blue if positive, red if negative

    def draw(self):
        pygame.draw.circle(gameDisplay, self.color, to_int(self.pos.coord()), 10)

    def __repr__(self):
        return f'PointCharge(x={self.x}, y={self.y})'

    def __str__(self):
        return str(self.pos)


class Particle():

    def __init__(self, x, y):
        self.pos = Vector(x, y)
        self.charge = 1
        self.acceleration = Vector(0, 0)
        self.velocity = Vector(0, 0)
        self.color = (255, 255, 255)

    def draw(self):
        gameDisplay.blit(particle_img, to_int(self.pos.coord()))


# point_charges = [PointCharge(100, 100, 5), PointCharge(500, 400, 4), PointCharge(250, 300, -1)]
# point_charges = [PointCharge(250, 300, -1), PointCharge(700, 300, -1), PointCharge(500, 300, 0.5)]
point_charges = [PointCharge(300 * i, 120, -1 if (i % 2) > 0 else 0.2) for i in range(10)]
point_charge_positions = [p.pos for p in point_charges]
point_charge_locations = [p.coord() for p in point_charge_positions]
point_charge_charges = np.array([p.charge for p in point_charges])
# particles = [Particle(150, 200), Particle(400, 200)]
particles = [Particle(i * 36, j * 21.6) for j in range(33) for i in range(33)]



def game_loop():
    while 1:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.draw.rect(gameDisplay, (0, 0, 0), [0, 0, 1280, 720])
        #Draw everything
        for point_charge in point_charges:
            point_charge.draw()
        time1 = 0.0
        particle_pos = np.array([particle.pos for particle in particles])
        particle_charges = np.array([particle.charge for particle in particles])
        raw_differences = (point_charge_positions - particle_pos[:, np.newaxis])
        # print(raw_differences)
        # print(raw_differences.shape)

        x_diff = np.vectorize(lambda obj: obj.x)(raw_differences)
        y_diff = np.vectorize(lambda obj: obj.y)(raw_differences)
        directions = np.dstack((x_diff, y_diff))
        distances = np.sum(directions ** 2, axis=2) ** 1.5
        distances[distances < 1000] = 1000
        # print(distances.shape)
        net_force = np.sum(np.multiply(raw_differences,
                                       np.divide(((point_charge_charges * particle_charges[:, np.newaxis]) * -K),
                                                 distances)), axis=1)

        #print(time1)
        #print(distance_time, directions_time, magnitudes_time, forces_time, net_force_time)
        for i, particle in enumerate(particles):
            particle.acceleration = net_force[i]
            particle.velocity += particle.acceleration/FPS
            particle.pos += particle.velocity
            particle.draw()
        pygame.display.update()
        clock.tick(FPS)

pygame.init()
game_loop()