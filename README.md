# Motion Detection in OpenCV
This reposirory implement an application in Python using OpenCV to detect a motion from a video stream, either from a video file or a webcam. The processing is done in real time. The accuracy of the detection could be controled with a threshold parameter

# Demo
## Video stream from the laptop webcam
![demo_real_time.gif](demo_real_time.gif)

## Video stream from a video file
![demo_fire.gif](demo_fire.gif)

# Usage
## Run the script (if you have OpenCV already installed)
python motion_detection.py

## Run the AI-lab and start your development
If you don't have OpenCV installed on your machine, you can use AI-lab, a complete development envirnement to run your computer vision application.

Simply install `docker-ce` and then run in the terminal:

``` bash
xhost +
docker run -it --rm 
--runtime=nvidia 
-v $(pwd):/workspace \
-w /workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
-p 8888:8888 -p 6006:6006 aminehy/ai-lab:latest
```
