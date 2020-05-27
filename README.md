python3 inference_mega_tensorflow.py

input


![image](./doc/demo.jpg)


output


![image](./doc/hell0_demo_tf_320x240_prepost.png)


Using ffmpeg to make side by side movie
ffmpeg -i video_from_images_KITTI.avi -i video_from_depths_KITTI.avi -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] -c:v libx264 -crf 23 -preset veryfast output.mp4 
