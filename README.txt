This project implements image processing algorithms (currently only skeletonization) on video files. It utilizes streams to load and process frames in parallel. 

Notes:
- Skeletonization algorithm was implemented during Module 9. LinAnnabel_Module9Assignment_Writeup is included for background.
- New work 
    - Extend to video frames
    - Add command line options for threshold and black/white setting


Requirements:
- install opencv   https://opencv.org/releases/
    I have 4.12.0 on Windows. Also needed to add to Path environment variable

GPU code in assignment.cu

To compile: make

To run: assignment.exe -i inputs/cat_looking.mp4 -m 0

args:
-i <input file>
-t <b/w threshold value>
-m <b/w object mode, where 0=black object on white background, 1=white object on black background>



Video sources:
cat_looking.mp4: https://www.pexels.com/video/video-of-black-cat-855401/

