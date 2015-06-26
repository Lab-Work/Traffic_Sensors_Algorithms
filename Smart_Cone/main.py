from SmartCone import SmartCone
import warnings

emulator = SmartCone(estimatorType='adaptiveThreshold')
#emulator.label()
#emulator.meanTempHist()
emulator.timeSeries()
#emulator.heatMap(fps=80,saveFig=True)

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
