# This sciprt contains useful ffmpeg commands for video generation

# Combine png files into mp4 video
ffmpeg -framerate 8.261 -pattern_type glob -i \
'visualization/sensor_status/*.png' -r 30 -vb 20M -pix_fmt yuv420p \
sensors_status.mp4

# Cut example video file from 00:00:15 for 00:09:50 to 00:10:05
# This will reduce video quality
ffmpeg -i input.mp4 -ss 00:00:03 -t 00:09:50 -async 1 -strict 2 cut.mp4

# Crop video to 1000 by 720 starting at pixel (140, 0)
ffmpeg -i cut.mp4 -filter:v "crop=1000:720:140:0" -c:a copy copy.mp4

# Scale video to 320 by 240
ffmpeg -i input.avi -vf scale=320:240 output.avi

# Vertically stacking two equal-width videos
ffmpeg -i copy.mp4 -i sensors_status.mp4 -filter_complex vstack demo.mp4
