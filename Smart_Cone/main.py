from SmartCone import SmartCone
import warnings

emulator = SmartCone('adaptiveThreshold')

while True:
    try:
        emulator.estimate()
        emulator.updateBuffer()
    except KeyboardInterrupt:
        break
    except:
        warnings.warn('Oops, something is not right...')
