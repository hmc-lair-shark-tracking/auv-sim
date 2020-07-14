"""
A wrapper class to represent the shark's state
    including range and bearing
    also a boolean variable do indicate if we have new reading
"""
class SharkState:
    def __init__(self, x, y, range_in, bearing_in, id):
        self.range = range_in
        self.x = x
        self.y = y
        self.id = id
        self.bearing = bearing_in

    def __repr__(self):
        return "[x=" + str(self.x) + ", y=" + str(self.y) + ", range= "+ str(self.range) +", bearing = " + str(self.bearing) + "]"

    def __str__(self):
        return "[x=" + str(self.x) + ", y=" + str(self.y) + ", range= "+ str(self.range) +", bearing = " + str(self.bearing) + "]"