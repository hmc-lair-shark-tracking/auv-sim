"""
A wrapper class to represent the shark's state
    including range and bearing
"""
class SharkState:
    def __init__(self, range_in, bearing_in):
        self.range = range_in
        self.bearing = bearing_in

    def __repr__(self):
        return "[" + str(self.range) + ", " + str(self.bearing) + "]"

    def __str__(self):
        return "[" + str(self.range) + ", " + str(self.bearing) + "]"