from SmartCone import SmartCone
import warnings

emulator = SmartCone(estimatorType='adaptiveThreshold')
#emulator.label()
#emulator.meanTempHist()
<<<<<<< HEAD
emulator.timeSeries('ultrasonic')
#emulator.timeSeries()
#emulator.heatMap(fps=80,saveFig=False)
=======
emulator.timeSeries()
#emulator.heatMap(fps=80,saveFig=True)
>>>>>>> 8453e30773606885b1672301e062b2fb1a80e4b1

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
