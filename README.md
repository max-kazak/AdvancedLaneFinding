Advanced Lane Finding
============================================

In this project, I wrote a software pipeline comprised of various Computer Vision techniques to identify the lane boundaries in a video as a prototype for autonomous car perception system responsible for lane navigation.

[![Result Video](report/youtube_thumbnail.jpg)](https://www.youtube.com/watch?v=IhAv3MpZY7k)

## Project Structure

Project has following directory structure:  

**.**  
├── **data**                 - folder with all data  
│   ├── **camera_cal**       - camera calibration images    
│   ├── **test_images**      - folder with raw test images  
│   ├── **videos**           - folder with raw videos  
│   ├── **output_images**    - here goes all processed test images  
│   └── **output_videos**    - here goes all processed videos  
├── **lanefinder**           - project's source code  
│   ├── **calibration.py**   - module with camera calibration related code  
│   ├── **detect.py**        - module with detection related code  
│   ├── **exceptions.py**    - module with custom exceptions  
│   ├── **paths.py**         - module with path constants  
│   ├── **pipeline.py**      - module with pipline related code  
│   ├── **processing.py**    - entry-point module that launches image/video processing  
│   ├── **thresholding.py**  - module with image thresholding related code  
│   ├── **transform.py**     - module with image transformation related code  
│   └── **utils.py**         - module with common-purpose functionality code  
├── **report**               - folder with report images  
├── **calibration.p**        - pickle file with saved calibration matrices  
└── **requirements.txt**     - project library requirements  

## Execution

To run the code yourself prepare Python3 environment by downloading this repository and installing necessary libraries:  

```
git clone https://github.com/max-kazak/AdvancedLaneFinding.git
cd AdvancedLaneFinding
pip3 install -r requirements.txt
```

After that you're ready to run the code:  

```
python3 lanefinder/processing.py
```


## Pipeline  

Following diagram shows key components of the software pipeline and flow of the data through them. Below the diagram is the detailed description of the pipline nodes.  

![Calibration>Thresholding>BirdviewTransformation>LaneDetection>LaneValidation>LaneSmoothing>CameraViewTransformation>ARProcessing](report/LaneFinderPipeline.png)  

### Camera Calibration  

Because no camera lens or sensor are perfect both of those components add distortions to the image they capture. Before proceeding to the lane detection these distortions have to be addressed first.  
In this project I'm using set of images capturing chessboard from various angles.  
![](report/calibration/distorted_image.png)  
By detecting corners of the chessboard it is possible to calculate calibration matrices.  
![](report/calibration/find_corners.png)  
Calculated calibration matrices are stored into pickle file *calibration.p* to be reused later during video processing. This way calibration process can be skipped to reduce processing time.  
After calibration process images and video frames can be undistorted using calibration matrices.  
![](report/calibration/undistorted_image.png)  

### Thresholding  

To build the model of the lane it is first necessary to segment-out pixels that belong to the lane lines. In this project I'm using gradient and color thresholding to create binary masks of the line pixels.  

#### Gradient thresholding  

Image gradients show how fast pixel intensity is changing in vertical and horizontal directions. Thus it is possible to detect edges and lines on the images.  
So to separate road lines I created binary mask that shows pixels which gradient magnitude and direction are within specified boundaries.  
![](report/test_images/001.jpg)![](report/thresholding/grad2.jpg)  
 
 This method produces well defined lines that are easy to use for creating line models. But one of the big downsides of such approach is if intensity of the line is low compared to the road it's very difficult to discern it using gradients.  
 
![](report/test_images/005.jpg)![](report/thresholding/grad6.jpg) 

#### Color thresholding  

To address shortcomings of the gradient thresholding I combined it with color thresholding. Color thresholding is the process of image segmentation based on the color of particular pixel.  
But due to changing conditions on the road (e.g. shadows) choosing colors in RGB color space is unreliable. Instead I used HSL color space that separates Hue and Saturation of the color from Lightness that changes the most in the shadows. Using thresholds based on Hue boaundaries and Saturation I achieved much more consistent lane lines separation.  

![](report/test_images/005.jpg)![](report/thresholding/color6.jpg) 


### Cut ROI

When looking for the lane we can make a rough guess where on the image it can be, we'll call it out region of interest (ROI). By eliminating all pixels outside of this region we can reduce processing time and reduce noise that can affect model calculation.  

### Bird-view perspective  

Bird-eye view persective (aka top view perspective) can be advantageous in many calculations. I'm using it here as well as it will significantly simplify mathematical models used to find lane lines.  

![](report/warp/img_orig1.jpg)![](report/warp/img_warped1.jpg)  


### Lane detection 

This pipeline node is responsible for choosing pixels that will belong to the lane lines, creating polynomial model of theses lines and creation of the Lane object that encopsulates both and add extra functionality such as line curvature calculation and car offset from the lane center.  

Lane pixels can be chosen either from scratch or using prior lane models.  
When looking for lane from scratch I first used histograms of the bottom part of the image to find initial bottom lane lines positions. It is done by finding two histogram peaks on left and right halfs of the image each. 

![](report/lanes/histogram2.jpg)

I used small windows around these initial positions to determine which pixels are close enough to it. If window covers enough masked pixels algorithm calculates new position for the windows drawn on top of the first ones. These process of sliding windows repeats until it reaches top of the image (in bird-eye view perspective).  

![](report/lanes/windows2.jpg)

Second approach takes advantage of the fact that lanes don't move much in adjacent frames. As such we can use prior lane line models to create area where new lines' pixels should be. All masked pixels that are covered by this area becomes part of the new Lane object.  

Finally after finding left and right lines pixel locations we can fit polynomial of second degree (x = Ay^2 + By + C) to it, which will become our line models. 

![](report/lanes/fitted_lines2.jpg)

### Lane Validation 

Even after carefully tuning parameters it is still difficult to avoid problems in some in all video frames. Lane Validation node is responsible to finding bad Lane objects after the fact and dealing with such problems. 

Lane validation is done by comparing two lane lines to each other and rejecting Lane object if say lines curve in different directions or if one line curves much sharper than the other. It also can compare new lane model to the model from the previous frame and again reject new model if it deviates from the prior one too much.  

If new Lane object didn't pass the validation check one of the following happens: 

 - prior Lane object is reused for current frame 
 - lane is detected from scratch if detection failed too many times
 
### Lane smoothing  

Changes in lane model can be small enough to pass validation check but still cause ragged estimation when estimated lane rapidly changes position in adjacent frames. To avoid it mean of the several past models is used for further processing.  

### Augmeted Reality

The final part of this project is to created augmented video with estimated lane projected on top of the road. 
This is done by drawing a lane on the empty bird-eye view image and changing perspective of this image back to camera view one. Then lane image is combined with undistorted frame and text with Lane object information is added on top. 

![](report/output_example.jpg)
