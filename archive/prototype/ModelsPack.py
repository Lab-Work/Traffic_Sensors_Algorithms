"""
@author: Fangyu Wu
@date: 07/27/15
@description: the class that stores a repository of prediction models for real
time vehicle detection and speed estimation.
"""

class Models:

    def __init__(self, cache):
        print "Initializing Models..."
        self.density = 'M'
        self.speed = 'M'
        self.athr = adaptiveThreshold(cache)
        self.gmix = gaussianMixure(cache)
        self.sensor_status = "on"
        from datetime import datetime
        start = datetime.now()

    def estimate(self, cache):
        if (self.speed == 'L'):
            if (self.density == 'L'):
                continue
            elif (self.density == 'M'):
                continue
            elif (self.density == 'H'):
                continue

        elif (self.speed == 'M'):
            if (self.density == 'L'):
                continue
            elif (self.density == 'M'):
                return method_MM(data)
            elif (self.density == 'H'):
                continue

        elif (self.speed == 'H'):
            if (self.density == 'L'):
                continue
            elif (self.density == 'M'):
                continue
            elif (self.density == 'H'):
                continue

    def method_MM(self, cache):
        data = cache[len(cache)-1]
        dist, temp, sensor_error = parse(data)
        estimate = []

        if (sensor_error):
            self.sensor_status = "error"
            estimate.append(['-', datetime.now()-start,
                             "SENSOR ERROR, RESTART"])
            return

        # output format:
        # timestamp, (flow), (density), vehicle count, (speed)
        athr_out, athr_error = self.athr.update(dist)
        gmix_out, gmix_error = self.gmix.update(temp)
        if (athr_error or gmix_error):
            if (athr_error and gmix_error):
                self.athr.reset(cache)
                self.gmix.reset(cache)
                estimate.append(['-', datetime.now()-start,
                                 "ALL MODELS REFRESH", data])
            elif (athr_error and not gmix_error):
                self.athr.reset(cache)
                estimate.append(['-', datetime.now()-start,
                                 "AT MODEL REFRESH", data])
            elif (not athr_error and gmix_error):
                self.gmix.reset(cache)
                estimate.append(['-', datetime.now()-start,
                                 "GM MODEL REFRESH", data])
        else:
            estimate.append(['+', datetime.now()-start,
                             athr_out, gmix_out, data])
        return estimate

class adaptiveThreshold:

    def __init__(self, cache):
        orbits = np.histogram(cache.data, bins=np.arange(0,200,5))
        mask = [0.25, 1, 0.25]
        levels = np.convolve(orbits[0], mask, "valid")
        levels_sorted = np.sort(levels)
        max_level = levels_sorted[-2]
        self.targets = 1 > levels/max_level > 0.4
        self.ground = levels == level_sorted[-1]

        self.status = False
        self.ispass = False
        self.ocur = 0
        self.count = 0

    def update(self, data):
        index = data/5
        self.ispass = self.targets[index]

        if (ispass):
            if (self.status == False):
                self.status = True
            else:
                self.ocur += 1
        else:
            if (self.status == False):
                continue
            elif (self.ocur > 2):
                self.count += 1
        return [self.count, False]

    def reset(self, cache):
        orbits = np.histogram(cache.data, bins=np.arange(0,200,5))
        mask = [0.25, 1, 0.25]
        levels = np.convolve(orbits[0], mask, "valid")
        levels_sorted = np.sort(levels)
        max_level = levels_sorted[-2]
        self.targets = 1 > levels/max_level > 0.4
        self.ground = levels == level_sorted[-1]

        self.status = False
        self.ispass = False
        self.ocur = 0
        self.count = 0

class gaussianMixture:

    def __init__(self, cache):
        continue

    def update(self, data):
        return [0, False]

    def reset(self, cache):
        continue
