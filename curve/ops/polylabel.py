from shapely.algorithms.polylabel import Cell
from shapely.geometry import LineString
from shapely.geos import TopologicalError
from heapq import heappush, heappop


class MyCell(Cell):
    def _dist(self, polygon):
        """leans to coordinates mean
        """
        inside = polygon.contains(self.centroid)
        distance = self.centroid.distance(LineString(polygon.exterior.coords))
        if inside:
            c_dist = self.centroid.distance(polygon.centroid)
            minx, miny, maxx, maxy = polygon.bounds
            cell_size = max(maxx - minx, maxy - miny, c_dist) + 1e-10
            distance = (distance ** 0.5 + cell_size ** 0.5) * (1.0 - c_dist / cell_size)
            return distance
        return -distance


def polylabel(polygon, tolerance=0.1):
    if not polygon.is_valid:
        raise TopologicalError('Invalid polygon')
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    cell_size = min(width, height)
    h = cell_size / 2.0
    cell_queue = []

    # First best cell approximation is one constructed from the centroid
    # of the polygon
    x, y = polygon.centroid.coords[0]
    best_cell = MyCell(x, y, 0, polygon)

    # Special case for rectangular polygons avoiding floating point error
    bbox_cell = MyCell(minx + width / 2.0, miny + height / 2, 0, polygon)
    if bbox_cell.distance > best_cell.distance:
        best_cell = bbox_cell

    # build a regular square grid covering the polygon
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            heappush(cell_queue, MyCell(x + h, y + h, h, polygon))
            y += cell_size
        x += cell_size

    # minimum priority queue
    while cell_queue:
        cell = heappop(cell_queue)

        # update the best cell if we find a better one
        if cell.distance > best_cell.distance:
            best_cell = cell

        # continue to the next iteration if we cant find a better solution
        # based on tolerance
        if cell.max_distance - best_cell.distance <= tolerance:
            continue

        # split the cell into quadrants
        h = cell.h / 2.0
        heappush(cell_queue, MyCell(cell.x - h, cell.y - h, h, polygon))
        heappush(cell_queue, MyCell(cell.x + h, cell.y - h, h, polygon))
        heappush(cell_queue, MyCell(cell.x - h, cell.y + h, h, polygon))
        heappush(cell_queue, MyCell(cell.x + h, cell.y + h, h, polygon))

    return best_cell.centroid