"""R-tree empaquetado para puntos de cualquier dimensión, usado por el benchmark."""
import heapq
import math


class Node:
    def __init__(self, bounds, children=None, points=None):
        self.bounds = bounds
        self.children = children or []
        self.points = points or []

    @property
    def is_leaf(self):
        return bool(self.points)


class MultiDimensionalRTree:
    def __init__(self, points, dimensions, capacity=32):
        self.dimensions = dimensions
        self.capacity = capacity
        self.page_reads = 0
        self.root = self._build(points, 0)

    def _bounds_for_points(self, points):
        return tuple(
            (min(point[1][axis] for point in points),
             max(point[1][axis] for point in points))
            for axis in range(self.dimensions)
        )

    def _build(self, points, depth):
        bounds = self._bounds_for_points(points)
        if len(points) <= self.capacity:
            return Node(bounds, points=points)

        axis = depth % self.dimensions
        ordered = sorted(points, key=lambda point: point[1][axis])
        group_count = min(self.capacity, math.ceil(len(ordered) / self.capacity))
        group_size = math.ceil(len(ordered) / group_count)
        children = []

        for start in range(0, len(ordered), group_size):
            group = ordered[start:start + group_size]
            children.append(self._build(group, depth + 1))

        return Node(bounds, children=children)

    def _box_distance(self, bounds, point):
        distance = 0.0
        for axis, (lower, upper) in enumerate(bounds):
            if point[axis] < lower:
                distance += (lower - point[axis]) ** 2
            elif point[axis] > upper:
                distance += (point[axis] - upper) ** 2
        return distance

    @staticmethod
    def _point_distance(first, second):
        return sum((a - b) ** 2 for a, b in zip(first, second))

    def radius_search(self, center, radius):
        self.page_reads = 0
        radius_squared = radius ** 2
        matches = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            self.page_reads += 1
            if self._box_distance(node.bounds, center) > radius_squared:
                continue

            if node.is_leaf:
                for point_id, coordinates in node.points:
                    if self._point_distance(coordinates, center) <= radius_squared:
                        matches.append(point_id)
            else:
                stack.extend(node.children)

        return matches

    def nearest_neighbors(self, center, k):
        self.page_reads = 0
        queue = [(self._box_distance(self.root.bounds, center), 0, self.root)]
        sequence = 1
        best = []

        while queue:
            distance, _, item = heapq.heappop(queue)
            if len(best) == k and distance > -best[0][0]:
                break

            if isinstance(item, Node):
                self.page_reads += 1
                if item.is_leaf:
                    for point_id, coordinates in item.points:
                        point_distance = self._point_distance(coordinates, center)
                        heapq.heappush(queue, (point_distance, sequence, (point_id, point_distance)))
                        sequence += 1
                else:
                    for child in item.children:
                        lower_bound = self._box_distance(child.bounds, center)
                        heapq.heappush(queue, (lower_bound, sequence, child))
                        sequence += 1
                continue

            point_id, point_distance = item
            candidate = (-point_distance, point_id)
            if len(best) < k:
                heapq.heappush(best, candidate)
            elif candidate > best[0]:
                heapq.heapreplace(best, candidate)

        return sorted(((point_id, -negative_distance) for negative_distance, point_id in best),
                      key=lambda result: (result[1], result[0]))
