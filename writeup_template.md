#**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### 1. Reflection

My pipeline consisted of 4 steps (each one divided into sub-steps).

#### Prepare image
First step is making sure that yellow color will be handled fine. To accomplish that I implemented change_color function which changes yellow to white. Next step is to convert image to gray scale. Then - applying Gaussian_blur (kernel_size=5) and finally canny edges (low_threshold=50, high_threshold=150).

![Prepare image](/files/pipeline_1.png)

#### Define regions of interest
For this task I decided that I need 2 regions (calculated based on image dimensions):
![Prepare image](/files/regions_of_interest.png)

#### Find and filter lines
To find lines I used Hough transform on the edges in both regions of interest.
Results contains many unsuitable data so I implemented function filter_by_slope which removes lines with unacceptable slopes.

#### Calculations
In order to draw red lines over lane lines I needed values at extreme points.
I decided to use simple linear regression for this task.

![Calculations and drawing lines](/files/pipeline_2.png)



###2. Potential shortcomings
It turned out that current implementation has problems with situations like:
* when road changes it's color (see challenge.mp4, there will be a glitch)
* when distance between lane lines is too long


###3. Possible improvements
* Better color-correction. I'm sure that road color issue mentioned above could be fixed
* Straight lines don't look good when road is turning. I think it is better to draw curves instead.
* Add parametrization connected to real world: for example, params I use to change regions of interest should be connected with camera position on the car.
