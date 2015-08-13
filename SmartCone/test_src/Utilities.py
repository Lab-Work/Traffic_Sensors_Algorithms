'''
@author: Fangyu Wu
@date:   07/23/15
@description:

    [Cache]: class stores data read from sensors and results computed by the
    model and output to ostream when requested.
'''

class Cache:
    def __init__(self, size):
        self.data = []
        self.capacity = size

    def size(self):
        return len(self.data)

    # push new measurement and estimate into cache and throw away the least recent
    def fetch(self, data):
        self.data.append(data)
        if (self.size() > self.capacity)
            self.data.pop(0)

    # clear the cache in case things mess up
    def clear():
        self.data = []
        self.size = 0

    # write back to hard disk
    def write_back():
        print "Writing back to hard drive"
        with open("blah.csv", 'a') as file:
            for data in self._data
                file.write("%s\n") % data
        self.clear()

    # output to i/o devices
    def display():
        print self.data[-1]
        continue
