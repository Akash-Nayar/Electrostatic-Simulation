import pygame, sys, math
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances

K = 9*10

display_width = 1280
display_height = 720


particle_img = pygame.image.load('images/ball.png')


FPS = 30

gameDisplay = screen=pygame.display.set_mode((display_width,display_height))
surface = pygame.Surface((display_width,display_height), pygame.SRCALPHA)

clock = pygame.time.Clock()


def to_int(tup):
    return tuple(map(int, tup))




# Define Vector class:
#I realize now that pygame already has its own Vecotr class but I didn't know this when I wrote this class so here it is.
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

    def __init__(self, x, y, charge, phase_shift):
        self.pos = Vector(x, y)

        self.charge = charge
        self.color = (0, 0, 255) if charge > 0 else (255, 0, 0,)  # Blue if positive, red if negative

    def draw(self):
        #self.pos = Vector(np.random.randint(display_width), np.random.randint(display_height))
        pygame.draw.circle(surface, self.color, to_int(self.pos.coord()), 10)

    def __repr__(self):
        return f'PointCharge(x={self.x}, y={self.y})'

    def __str__(self):
        return str(self.pos)


class Particle():

    def __init__(self, x, y):
        self.pos = Vector(x, y)
        self.charge = 1
        self.alpha = 255
        self.acceleration = Vector(0, 0)
        self.velocity = Vector(0, 0)
        self.color = (255, 255, 255, self.alpha)

    def draw(self):
        #gameDisplay.blit(particle_img, to_int(self.pos.coord()))25
        self.color = (255, 255, 255, int(self.alpha))
        #print(self.color)
        pygame.draw.circle(surface, self.color, to_int(self.pos.coord()), 5)

    def draw_velocity(self):
        pygame.draw.line(gameDisplay, (0,0,255), to_int(self.pos.coord()), to_int((self.pos+self.velocity*5).coord()), 2)

    def draw_acceleration(self):
        pygame.draw.line(gameDisplay, (255,0,0), to_int(self.pos.coord()), to_int((self.pos+self.acceleration*5).coord()), 2)







def simulation_loop():
    tick = 0
    # point_charges = [PointCharge(100, 100, 5), PointCharge(500, 400, 4), PointCharge(250, 300, -1)]
    # point_charges = [PointCharge(250, 300, -1, 0.4), PointCharge(700, 300, -1, 0.1), PointCharge(500, 300, 0.5, 3)]
    num_point_charges = 5
    point_charges = [
        PointCharge(display_width / 2, display_height / 2, -1 if (i % 2) > 0 else 0.2, 2 * math.pi / num_point_charges)
        for i in range(num_point_charges)]
    for point_charge in point_charges:
        print(point_charge.pos)
    # particles = [Particle(150, 200), Particle(400, 200)]
    particles = [Particle(i * 36, j * 21.6) for j in range(33) for i in range(33)]
    while 1:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.draw.rect(surface, (0, 0, 0), [0, 0, 1280, 720])
        point_charge_positions = [p.pos for p in point_charges]
        point_charge_locations = [p.coord() for p in point_charge_positions]
        point_charge_charges = np.array([p.charge for p in point_charges])
        time1 = 0.0
        particle_pos = np.array([[list(particle.pos.coord())] for particle in particles])
        particle_charges = np.array([particle.charge for particle in particles])

        raw_differences = point_charge_locations - particle_pos

        x_diff = raw_differences[:, :, 0]
        y_diff = raw_differences[:, :, 1]

        distances = np.divide(y_diff, np.sin(np.arctan2(y_diff, x_diff))) ** 1.5

        magnitudes = np.divide(((point_charge_charges * particle_charges.reshape(len(particle_charges), 1)) * -K),
                               distances).reshape(len(particles), len(point_charges), 1)

        net_force = np.sum(np.multiply(raw_differences, magnitudes), axis=1)
        net_force_magnitudes = np.sum(np.abs(magnitudes), axis=1)
        net_force_magnitudes /= np.max(np.abs(net_force_magnitudes), axis=0)
        opacities = net_force_magnitudes ** 1.5 *255
        np.max(opacities)

        #print(time1)
        #print(distance_time, directions_time, magnitudes_time, forces_time, net_force_time)
        # Draw everything
        for point_charge in point_charges:
            point_charge.draw()
        for i, particle in enumerate(particles):
            particle.acceleration = Vector(net_force[i,0], net_force[i,1])
            particle.velocity += particle.acceleration/FPS
            particle.pos += particle.velocity
            #particle.alpha = opacities[i]
            particle.draw()
            #particle.draw_velocity()
            #particle.draw_acceleration()
        tick += 1
        gameDisplay.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(FPS)

pygame.init()
simulation_loop()