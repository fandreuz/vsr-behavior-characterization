from collections import namedtuple

VSR = namedtuple("VSR", ["shape", "training_terrain", "seed"])

# represents a cluster of VSR
class Cluster:
    def __init__(self, name):
        self._name = name
        self._items = set()

    def add(self, vsr):
        self._items.add(vsr)

    def difference(self, other):
        return self._items - other._items

    def intersect(self, other):
        return self._items.intersection(other._items)

    def size(self):
        return len(self._items)

    def __ne__(self, other):
        return len(self.difference(other)) > 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "[" + ";".join(map(str, self._items)) + "]\n"
