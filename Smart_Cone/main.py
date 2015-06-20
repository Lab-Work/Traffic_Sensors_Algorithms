from SmartCone import SmartCone
import warnings

emulator = SmartCone(estimatorType='adaptiveThreshold')
emulator.label()
#emulator.meanTempHist()
emulator.timeSeries()

'''
while True:
    try:
        emulator.estimate()
        emulator.update()
    except KeyboardInterrupt:
        break
    except:
        warnings.warn('Oops, something is not right...')
'''
