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

    def __hash__(self):
        return hash(str(self.x) + str(self.y))
    
    def __eq__(self, node):
        return self.x == node.x and self.y == node.y

#=============================================================

# GLOBALS
w = []
cities = []
N = 0
inf = float("inf")
MAX_X, MIN_X, MAX_Y, MIN_Y, WIDTH, HEIGHT, MAX_VALUE = -inf, inf, -inf, inf, inf, inf, inf
BMU_cache = {}
STEPS = 1000
BMU_NEIGHBOURHOOD = 10
EXP_MOD = float(STEPS / 4.0)

def init():
    global w, cities, N, MAX_VALUE
    cities = loadCities("uruguay")
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
            for j in range(1,BMU_NEIGHBOURHOOD):
                neighbour = current_node.nxt
                distance = euclidean_distance(neighbour.x, city[0], neighbour.y, city[1])
                if distance < min_distance:
                    min_distance = distance
                    BMU = neighbour
                current_node = neighbour
            for j in range(1,BMU_NEIGHBOURHOOD):
                neighbour = current_node.prev
                distance = euclidean_distance(neighbour.x, city[0], neighbour.y, city[1])
                if distance < min_distance:
                    min_distance = distance
                    BMU = neighbour
                current_node = neighbour
            BMU_cache[str(city[0])+str(city[1])] = BMU

        current_node = BMU
        updateNode(BMU,city, count, distance)

        for j in range(1,radius(count)):
            neighbour = current_node.nxt
            distance = j

            if distance < radius(count):
                updateNode(neighbour,city,count, distance)
            current_node = neighbour

        current_node = BMU
        for j in range(1,radius(count)):
            neighbour = current_node.prev
            distance = j

            if distance < radius(count):
                updateNode(neighbour,city,count, distance)
            current_node = neighbour

def find_path():
    path = []
    node_to_cities = {}
    for city in cities:
        node = BMU_cache[str(city[0])+str(city[1])]

        if node not in node_to_cities.keys():
            node_to_cities[node] = []
        node_to_cities[node].append(city)

    current_node = node_to_cities.keys()[0]
    for i in range(len(w)):
        if current_node in node_to_cities.keys():
            towns = node_to_cities[current_node]
            if len(path) > 0:
                temp_map = {}
                for town in towns:
                    distance = euclidean_distance(town[0], path[-1][0], town[1], path[-1][1])
                    temp_map[distance] = town
                sorted_keys = temp_map.keys()
                sorted_keys.sort()
                for key in sorted_keys:
                    path.append(temp_map[key])
            else:
                path.extend(towns)
        current_node = current_node.nxt
    path.append(path[0])
    return np.array(path)

def total_distance(path):
    total_distance = 0
    for i in range(len(path)-1):
        city1 = path[i]
        city2 = path[i+1]
        distance = euclidean_distance(city1[0], city2[0],city1[1],city2[1])
        total_distance += distance
    return total_distance * MAX_VALUE

def updateNode(node, city, count, distance):
    node.x += learning_function(count) * neighbourhood_function(distance) * (city[0] - node.x)
    node.y += learning_function(count) * neighbourhood_function(distance) * (city[1] - node.y)

def learning_function(count):
    #return 0.80 #Constant
    #return 1.0 - float(count / STEPS) # Linear
    return math.exp(float(float(-count)/EXP_MOD)) # exp


def radius(count):
    #return min(len(w) / 10, 50) # Constant
    start_radius = min(len(w) / 10,50)
    #return int(start_radius - float(float(count) / float(STEPS)) * start_radius) # Linear
    return int(start_radius * math.exp(float(float(-count)/EXP_MOD))) # exp

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
    list.append(list[0])
    return np.array(list)

def main():
    #pr = cProfile.Profile()
    #pr.enable()


    init()

    plot(nodes_to_data(w), 30)
    for i in range(STEPS):
        SOM(i)
        if i == 500:
            plot(nodes_to_data(w),30)
        if i % 100 == 0:
            print i
            print radius(i)
    plot(nodes_to_data(w), 30)
    path = find_path()
    print total_distance(path)
    plot(path,30)

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
