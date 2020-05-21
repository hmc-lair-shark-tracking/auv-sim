"""
A wrapper class to represent the shark's state
    including range and bearing
    also a boolean variable do indicate if we have new reading
"""
class SharkState:
    def __init__(self, range_in, bearing_in, id):
        self.range = range_in
        self.bearing = bearing_in
        self.newData = False
        self.id = id

    def __repr__(self):
        return "[" + str(self.range) + ", " + str(self.bearing) + "]"

    def __str__(self):
        return "[" + str(self.range) + ", " + str(self.bearing) + "]"