import pygame, sys, math
import numpy as np
import time
import numpy as np
from scipy import interpolate, stats
from noise import pnoise2
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances

K = 9*10

display_width = 1280
display_height = 720


particle_img = pygame.image.load('images/ball.png')

FPS = 30

gameDisplay = screen=pygame.display.set_mode((display_width,display_height), pygame.SRCALPHA)
surface = pygame.Surface((display_width,display_height), pygame.SRCALPHA)

clock = pygame.time.Clock()


def to_int(tup):
    return tuple(map(int, tup))


def rough_distance(tup1, tup2):
    return abs(tup1[0] - tup2[0]) < 2 and abs(tup1[1] - tup2[1]) < 2

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

steps = 100
future_points = 3

#This code is from: https://engineeredjoy.com/blog/perlin-noise/
def perlin_array(shape=(display_width, display_height), scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):

    if not seed:
        seed = np.random.randint(0, 100)

    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)
    max_arr = np.max(arr)
    min_arr = np.min(arr)
    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr

particles_x = 50
particles_y = 33
red_noise = perlin_array(shape=(particles_x, particles_y))
green_noise = perlin_array(shape=(particles_x, particles_y))
blue_noise = perlin_array(shape=(particles_x, particles_y))
border_threshold = 100

# Define PointCharge class
class PointCharge():

    def __init__(self, x, y, charge, phase_shift):
        self.step = 0
        self.pos = Vector(x, y)
        self.next_destinations = [[np.random.randint(display_width+2*border_threshold)-border_threshold, np.random.randint(display_height+2*border_threshold)-border_threshold] for _ in range(future_points)]
        self.charge = charge
        self.color = (0, 0, 255) if charge > 0 else (255, 0, 0,)  # Blue if positive, red if negative
        self.radius = abs(int(self.charge))
        x, y = zip(*self.next_destinations)
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        f, u = interpolate.splprep([x, y], s=0, per=True)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, steps*future_points), f)
        self.path = list(zip(xnew, ynew))

    def draw(self):

        #If we're about to hit our second to last point
        if self.step == (future_points-1)*steps:
            #print('this true')

            #Generate new points
            self.next_destinations = [self.path[self.step]] + [[np.random.randint(display_width+2*border_threshold)-border_threshold, np.random.randint(display_height+2*border_threshold)-border_threshold] for _ in range(future_points)]
            #print(self.next_destinations)
            x, y = zip(*self.next_destinations)
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
            current = self.path[self.step]
            f, u = interpolate.splprep([x, y], s=0, per=True)
            xnew, ynew = interpolate.splev(np.linspace(0, 1, steps * future_points), f)
            temp = list(zip(xnew, ynew))
            index = list(map(lambda tup1: rough_distance(tup1, current), temp)).index(True)
            self.path = temp[index:]
            #print(current, self.path[0])
            self.step = 0
        #print(self.path[self.step])
        self.pos = Vector(self.path[self.step][0], self.path[self.step][1])

        #self.pos = Vector(np.random.randint(display_width), np.random.randint(display_height))
        pygame.draw.circle(surface, self.color, to_int(self.pos.coord()), self.radius)
        self.step += 1

    def __repr__(self):
        return f'PointCharge(x={self.x}, y={self.y})'

    def __str__(self):
        return str(self.pos)


class Particle():

    def __init__(self, x, y, index):
        self.origin = Vector(x, y)
        self.pos = Vector(x, y)
        self.charge = 1
        self.alpha = 0
        self.acceleration = Vector(0, 0)
        self.velocity = Vector(0, 0)
        self.index_x, self.index_y = index
        self.color = [int(red_noise[self.index_x][self.index_y]*255), int(green_noise[self.index_x][self.index_y]*255), int(blue_noise[self.index_x][self.index_y]*255), self.alpha]


    def set_alpha(self, new_alpha):
        self.alpha = int(new_alpha)

    def draw(self):
        #gameDisplay.blit(particle_img, to_int(self.pos.coord()))25
        self.color[3] = 255-int(self.alpha)
        #print('my alpha, ',self.alpha)
        #self.color = (255, 255, 255, 255)
        #print(self.color)
        pygame.draw.circle(surface, self.color, to_int(self.pos.coord()), 3)

    def draw_velocity(self):
        # Draw Vector that describes Particle's velocity
        pygame.draw.line(surface, (0,0,255), to_int(self.pos.coord()), to_int((self.pos+self.velocity*5).coord()), 2)

    def draw_acceleration(self):
        #Draw Vector that describes Particle's accleration
        pygame.draw.line(surface, (255,0,0), to_int(self.pos.coord()), to_int((self.pos+self.acceleration*5).coord()), 2)

    def draw_trail(self):
        # Draw Particle's graphics trail
        self.color[3] = 255-int(self.alpha)
        pygame.draw.line(surface, self.color, to_int(self.pos.coord()), to_int((self.pos-self.acceleration/5).coord()), 2)


def simulation_loop():
    global partciles_x, particles_y
    tick = 0
    num_point_charges = 25
    charges = np.random.randn(num_point_charges) * 5

    point_charges = [
        PointCharge(np.random.randint(display_width), np.random.randint(display_height), charges[i], 2 * math.pi / num_point_charges)
        for i in range(num_point_charges)]
    #particles = [Particle(150, 200), Particle(400, 200)]
    particles = [Particle(i * display_width/particles_x, j * display_height/particles_y, (i,j)) for j in range(particles_y) for i in range(particles_x)]
    while 1:

        #Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        point_charge_positions = [p.pos for p in point_charges]
        point_charge_locations = [p.coord() for p in point_charge_positions]
        point_charge_charges = np.array([p.charge for p in point_charges])
        particle_pos = np.array([[list(particle.pos.coord())] for particle in particles])
        particle_charges = np.array([particle.charge for particle in particles])

        raw_differences = point_charge_locations - particle_pos

        x_diff = raw_differences[:, :, 0]
        y_diff = raw_differences[:, :, 1]
        y_diff[y_diff == 0] = 1

        distances = np.divide(y_diff, np.sin(np.arctan2(y_diff, x_diff))) ** 1.5
        distances[distances<100] = 100

        magnitudes = np.divide(((point_charge_charges * particle_charges.reshape(len(particle_charges), 1)) * -K),
                               distances).reshape(len(particles), len(point_charges), 1)

        net_force = np.sum(np.multiply(raw_differences, magnitudes), axis=1)
        net_force_magnitudes = np.sum(np.abs(magnitudes), axis=1).flatten()
        #net_force_magnitudes /= np.max(np.abs(net_force_magnitudes), axis=0)
        #print(distances)
        thresholds = ((np.interp(net_force_magnitudes, (net_force_magnitudes.min(), net_force_magnitudes.max()), (0, 1))) ** 2) * 255
        log_magnitudes = np.log(net_force_magnitudes)
        transform = stats.boxcox(net_force_magnitudes)[0]
        opacities = ((np.abs((np.interp(transform, (transform.min(), transform.max()), (-1, 1)))))) * 255 + 127.5
        #print(np.min(opacities), np.max(opacities), np.median(opacities), np.mean(opacities))
        #print(opacities)
        #print(max(opacities), min(opacities))
        #print(opacities[opacities<100].shape)
        #print(opacities[0])
        #plt.scatter(net_force_magnitudes, opacities)
        #plt.plot(xint, yint)
        #plt.show()
        #break


        # Redraw frame
        pygame.draw.rect(surface, (0, 0, 0, 255), [0, 0, 1280, 720])
        pygame.draw.rect(gameDisplay, (0, 0, 0), [0, 0, 1280, 720])



        for i, particle in enumerate(particles):
            particle.acceleration = Vector(net_force[i,0], net_force[i,1])
            #print(opacities[i])
            particle.set_alpha(opacities[i] if opacities[i] <= 255 else 255)
            if thresholds[i] < 15:
                particle.velocity = (particle.origin - particle.pos)/5
            else:
                particle.velocity += particle.acceleration/FPS
            particle.pos += particle.velocity/5
            #print(particle.alpha)
            particle.draw()
            particle.draw_trail()
            #particle.draw_velocity()
            #particle.draw_acceleration()
        for point_charge in point_charges:
            point_charge.draw()
        tick += 1
        gameDisplay.blit(surface, (0, 0))
        pygame.display.update()

        clock.tick(FPS)








pygame.init()
simulation_loop()