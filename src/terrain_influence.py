import csv
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

aggregate = True
# t if terrain, s if shape
dominant_key = 's'

dc = {}

shapes = set()
terrains = set()
clusters = set()

primary_key = terrains if dominant_key == 't' else shapes
secondary_key = shapes if dominant_key == 't' else terrains

with open('dataset/clusters.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for t,s,*l in reader:
        shapes.add(s)
        terrains.add(t)
        clusters.update(l)

        key = (t,s) if dominant_key == 't' else (s,t)

        dc[key] = list(map(int, l))

# we aggregate the lists of clusters for each shape (for a single terrain)
if aggregate:
    for k in primary_key:
        l2 = []
        for k2 in secondary_key:
            l2.extend(dc[k,k2])
            del dc[(k,k2)]
        dc[k] = l2

frq = {}
for key in primary_key if aggregate else product(primary_key, secondary_key):
    data = dc[key]

    f = [0 for _ in clusters]
    for i in data:
        f[i] += 1

    frq[key] = f

print(frq)


