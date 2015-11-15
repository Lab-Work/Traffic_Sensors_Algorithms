ffmpeg -framerate 8.261 -pattern_type glob -i \
'visualization/sensor_status/*.png' -r 30 -vb 20M  -pix_fmt yuv420p \
sensors_status.mp4
