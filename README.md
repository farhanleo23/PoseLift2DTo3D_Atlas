# Pose-Lifting 2D to 3D :

This project aims to implement a 3D human pose estimation framework on a Atlas 200DK board to obtain inference results on traffic control videos. The implementation is based on the VideoPose3D framework as described in the paper : 

Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3D human pose estimation in video with temporal convolutions and semi-supervised training. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

The model is trained on Human3.6M dataset and converted to offline format (.om) as supported by the Atlas board. The application has the capability to do accurate pose lifting from 2D key points to 3D joint locations and render that as a visual output. To be precise, given a custom video and its corresponding 2D key points, the framework is able to predict the 3D pose and output the visual result accurately. 

# Testing:

To test the above framework, run:

python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 --render --viz-subject input_video_file_name --viz-action custom --viz-camera 0 --viz-video path/to/custom_vide/input --viz-output path/to/output_file_directory --viz-size 6

eg: python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 --render --viz-subject right_turn.mp4 --viz-action custom --viz-camera 0 --viz-video /home/HwHiAiUser/HIAI_PROJECTS/videopose3d_om/custom_video_input/right_turn.mp4 --viz-output /home/HwHiAiUser/HIAI_PROJECTS/videopose3d_om/custom_video_output/right_turn.mp4 --viz-size 6

For detailed description of the command line arguments - common/arguments.py 

# Note: 

i] The application is capable of lifting a 2D pose to 3D pose. Hence a 2D keypoint detector like Keypoint-RCNN from torchvision or Detectron2 framework has to be used to    detect the 2D key points in the input video. 
      
ii] The input video has to be interpolated to have 50 fps for best results. 
    eg: ffmpeg -i stop.mp4 -filter:v "minterpolate=fps=50" -c:a copy stop.mp4
 
