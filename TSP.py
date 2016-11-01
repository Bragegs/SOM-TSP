import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cProfile, pstats, StringIO


class Node(object):
    def __init__(self, x, y):
        self.prev = None
        self.nxt = None
        self.x = x
        self.y = y

    def set_prev(self, prev):
        self.prev = prev

    def set_next(self, nxt):
        self.nxt = nxt

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

#=============================================================

# GLOBALS
w = []
cities = []
N = 0
inf = float("inf")
MAX_X, MIN_X, MAX_Y, MIN_Y, WIDTH, HEIGHT, MAX_VALUE = -inf, inf, -inf, inf, inf, inf, inf
BMU_cache = {}
STEPS = 1000

def init():
    global w, cities, N, MAX_VALUE
    cities = loadCities("western-sahara")
    MAX_VALUE = np.array(cities).max()
    cities = normalize(cities)
    N = len(cities)
    w = generateRandomNodes(N * 2)


def generateRandomNodes(n):
    nodes = []
    center_x = MIN_X + WIDTH / 2
    center_y = MIN_Y + HEIGHT / 2
    radius = min(WIDTH, HEIGHT) / 6
    for i in range(n):
        x = center_x + math.cos(2*math.pi/n*i)*radius
        y = center_y + math.sin(2*math.pi/n*i)*radius

        node = Node(x / MAX_VALUE, y / MAX_VALUE)

        if len(nodes) > 0:
            prev = nodes[-1]
            node.set_prev(prev)
            prev.set_next(node)

        nodes.append(node)

    first = nodes[0]
    last = nodes[-1]
    first.set_prev(last)
    last.set_next(first)

    return nodes


def normalize(list):
    return np.array(list) / MAX_VALUE


def SOM(count):
    global w, BMU_cache
    indexes = [x for x in range(len(cities))]
    random.shuffle(indexes)
    for index in indexes:
        city = cities[index]
        BMU = None
        if count == 0:
            min_distance = inf

            for node in w:
                distance = euclidean_distance(node.x, city[0], node.y, city[1])

                if distance < min_distance:
                    min_distance = distance
                    BMU = node
            BMU_cache[str(city[0])+str(city[1])] = BMU
        else:
            prev_BMU = BMU_cache[str(city[0])+str(city[1])]
            min_distance = euclidean_distance(prev_BMU.x, city[0], prev_BMU.y, city[1])
            current_node = prev_BMU
            BMU = prev_BMU
            for j in range(1,10):
                neighbour = current_node.nxt
                distance = euclidean_distance(neighbour.x, city[0], neighbour.y, city[1])
                if distance < min_distance:
                    min_distance = distance
                    BMU = neighbour
                current_node = neighbour
            for j in range(1,10):
                neighbour = current_node.prev
                distance = euclidean_distance(neighbour.x, city[0], neighbour.y, city[1])
                if distance < min_distance:
                    min_distance = distance
                    BMU = neighbour
                current_node = neighbour
            BMU_cache[str(city[0])+str(city[1])] = BMU



        current_node = BMU
        BMU.x += learning_function(count) * neighbourhood_function(distance) * (city[0] - BMU.x)
        BMU.y += learning_function(count) * neighbourhood_function(distance) * (city[1] - BMU.y)
        for j in range(1,radius(count)):
            neighbour = current_node.nxt
            distance = j

            if distance < radius(count):
                neighbour.x += learning_function(count) * neighbourhood_function(distance) * (city[0] - neighbour.x)
                neighbour.y += learning_function(count) * neighbourhood_function(distance) * (city[1] - neighbour.y)
            current_node = neighbour

        current_node = BMU
        for j in range(1,radius(count)):
            neighbour = current_node.prev
            distance = j

            if distance < radius(count):
                neighbour.x += learning_function(count) * neighbourhood_function(distance) * (city[0] - neighbour.x)
                neighbour.y += learning_function(count) * neighbourhood_function(distance) * (city[1] - neighbour.y)
            current_node = neighbour

def learning_function(count):
    #return 0.99 Constant
    #return 1.0 - float(count / STEPS) # Linear
    return math.exp(-count/100) # exp


def radius(count):
    #return len(w) / 10 # Constant
    start_radius = len(w) / 10
    #return int(start_radius - float(float(count) / float(STEPS)) * start_radius) # Linear
    return int(start_radius * math.exp(-count/100)) # exp

def neighbourhood_function(distance):
    return 1.0 / float(distance + 1.0) # Exp


def euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def loadCities(city):
    global MAX_X, MIN_X, MAX_Y, MIN_Y, WIDTH, HEIGHT
    file = open('cities/' + city + '.txt', 'r')
    c = []
    for line in file:
        coordinate = line.strip("\n").split(" ")
        x, y = float(coordinate[0]), float(coordinate[1])
        if x > MAX_X:
            MAX_X = x
        if x < MIN_X:
            MIN_X = x
        if y > MAX_Y:
            MAX_Y = y
        if y < MIN_Y:
            MIN_Y = y
        c.append(np.array([x, y]))
    WIDTH = MAX_X - MIN_X
    HEIGHT = MAX_Y - MIN_Y
    return np.array(c)

def nodes_to_data(nodes):
    list = []
    for node in nodes:
        list.append(np.array([node.x, node.y]))
    return np.array(list)

def main():
    #pr = cProfile.Profile()
    #pr.enable()


    init()

    plot(nodes_to_data(w), 30)
    for i in range(STEPS):
        SOM(i)
        if i % 301 == 0:
            plot(nodes_to_data(w), 30)
    plot(nodes_to_data(w), 30)

    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'tottime'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()


def plot(data, code):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    red = plt.get_cmap('jet')(230)
    ax.scatter(cities[:, 0], cities[:, 1], color=red)

    color = plt.get_cmap('jet')(code)
    ax.scatter(data[:, 0], data[:, 1], color=color)
    ax.plot(data[:, 0], data[:, 1], color=color)

    plt.show()


if __name__ == "__main__":
    main()
