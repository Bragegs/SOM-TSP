import numpy as np
import random
import math

# GLOBALS
w = []
cities = []
inf = float("inf")
MAX_X, MIN_X, MAX_Y, MIN_Y = -inf, inf, -inf, inf


def init():
    global w, cities
    cities = normalize(loadCities("western-sahara"))
    w = randomizeWeight(len(cities))
    print cities


def randomizeWeight(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = random.random()
    return matrix


def generateRandomInputVector(n):
    input_vectors = []
    for i in range(n):
        x = random.randrange(MIN_X, MAX_X, 0.0001)
        y = random.randrange(MIN_Y, MAX_Y, 0.0001)
        input_vectors.append([x, y])
    return np.array(input_vectors)


def normalize(list):
    max_value = list.max()
    return list / max_value


def SOM():
    global w
    s = 0
    n = len(cities)
    input_vectors = generateRandomInputVector(n)
    for i in range(n):
        min_distance = inf
        BMU = None
        node = input_vectors[i]
        for j in range(n):
            city = cities[j]
            distance = euclidean_distance(node[0], city[0], node[1], city[1])
            if distance < min_distance:
                min_distance = distance
                BMU = city
        for j in range(n):
            city = cities[j]
            distance = euclidean_distance(node[0], city[0], node[1], city[1])
            if distance < neighbourhood_function():
                current_weight = w[i][j]
                # TODO: fix this -> D(t)
                w[i][j] = current_weight + learning_function() * ()
    s += 1


def learning_function():
    return 0.99

def neighbourhood_function():
    return 0.1


def euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def loadCities(city):
    global MAX_X, MIN_X, MAX_Y, MIN_Y
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
        c.append([x, y])
    return np.array(c)


def main():
    init()


if __name__ == "__main__":
    main()
