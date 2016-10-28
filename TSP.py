import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath


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


def init():
    global w, cities, N, MAX_VALUE
    cities = loadCities("western-sahara")
    MAX_VALUE = np.array(cities).max()
    cities = normalize(cities)
    N = len(cities)
    w = generateRandomNodes(N * 2)


def generateRandomNodes(n):
    nodes = []
    for i in range(n):
        x = random.uniform(MIN_X + WIDTH / 4, MAX_X - WIDTH / 4)
        y = random.uniform(MIN_Y + HEIGHT / 4, MAX_Y - HEIGHT / 4)

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
    global w
    for city in cities:
        min_distance = inf
        BMU = None
        for node in w:
            distance = euclidean_distance(node.x, city[0], node.y, city[1])
            if distance < min_distance:
                min_distance = distance
                BMU = node
        for j in range(len(w)):
            neighbour = w[j]
            distance = neighbour_count(BMU, neighbour)

            if distance < radius():
                w[j].x += learning_function(count) * neighbourhood_function(BMU, neighbour) * (city[0] - w[j].x)
                w[j].y += learning_function(count) * neighbourhood_function(BMU, neighbour) * (city[1] - w[j].y)
                #print "L", learning_function(count)
                #print "N", neighbourhood_function(BMU, neighbour)
                #print "count", count



def learning_function(count):
    return 1.0 - float(count / 2000)


def radius():
    return len(w) / 20

def neighbourhood_function(BMU, neighbour):
    return 1.0 / float(neighbour_count(BMU, neighbour) + 1.0)


def euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def neighbour_road_length(n1, n2):
    current_node = n1.prev
    prev_distance = 0
    next_distance = 0

    for i in range(len(w)):
        prev_distance += euclidean_distance(n1.x , current_node.x, n1.y, current_node.y)

        if current_node.x == n2.x and current_node.y == n2.y:
            break
        else:
            current_node = current_node.prev

    current_node = n1.nxt
    for i in range(len(w)):
        next_distance += euclidean_distance(n1.x , current_node.x, n1.y, current_node.y)

        if current_node.x == n2.x and current_node.y == n2.y:
            break
        else:
            current_node = current_node.nxt
    return min(prev_distance, next_distance)

def neighbour_count(n1, n2):
    if n1.x == n2.x and n1.y == n2.y:
        return 0

    current_node = n1.prev
    prev_count = 0
    next_count = 0

    for i in range(len(w)):

        if current_node.x == n2.x and current_node.y == n2.y:
            prev_count = i + 1
            break
        else:
            current_node = current_node.prev

    current_node = n1.nxt
    for i in range(len(w)):

        if current_node.x == n2.x and current_node.y == n2.y:
            next_count = i + 1
            break
        else:
            current_node = current_node.nxt
    return min(prev_count, next_count)


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
    init()

    plot(nodes_to_data(w), 10000)
    for i in range(2000):
        SOM(i)
        #if i % 300 == 0:
    plot(nodes_to_data(w), 30)


def plot(data, code):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    red = plt.get_cmap('jet')(230)
    ax.scatter(cities[:, 0], cities[:, 1], color=red)

    color = plt.get_cmap('jet')(code)
    ax.scatter(data[:, 0], data[:, 1], color=color)

    plt.show()


if __name__ == "__main__":
    main()
