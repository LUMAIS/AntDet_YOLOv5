# AntDet_YOLOv5
Ants and their Activities (Trophallaxis) Detection using YOLOv5 based on PyTorch.

Author: Artem Lutov &lt;&#108;u&#97;&commat;&#108;ut&#97;n.ch&gt;, Valentyna Pryhodiuk &lt;v&#112;ryhodiuk&commat;lumais&#46;&#99;om&gt;  
License: [Apache License, Version 2](www.apache.org/licenses/LICENSE-2.0.html)  
Organizations: [UNIFR](https://www.unifr.ch), [Lutov Analytics](https://lutan.ch/), [LUMAIS](http://lumais.com)

## Requirements
Install Python bindings:
```sh
$ python3 -m pip -r install requirements.txt 
```

## :crystal_ball: orbAnalysis
### Descriprion
The script displays two successive consecutive frames, where annotated objects are surrounded by bounding boxes.
Annotations should follow the [Labelbox style](https://docs.labelbox.com/reference/bounding-box)
for the video annotations exporting. To simplify the ID tracking, objects with same ID are connected with the lines.
Possibilities of the script:
#### 🔴 Connecting lines
- Press `1` to turn on the mode where each line is visualised.
Line and bbox gets highlighted if the user hover the mouse over the object.
- Press `2` to turn on the mode where only hovered objects have their lines displayed and highlighted.
- Press `3` to stop displaying the connecting lines.
#### 🟠 Feature IDs
 - `LB` click inside the object to start featureId changing mode. Borders become dashed.
   - If you want to **exchange** the featureId with another object,
   do the `LB` click inside that object on the **left frame**.
   Exchange automatically performs on every consecutive slide starting with the right frame.
   - If you want to interrupt the process and **perform the featureId changing mode
   on another object**, do the `LB` click inside that object on the **right frame**.
 - `LB` click outside the object:
   - On the right frame to interrupt featureId changing mode.
   - On the left frame nothing would happen.
#### 🟡 ROI drawing mode
Script allows to draw/display not more than one ROI. 
1. To allow ROI drawing mode press `d` 
2. To draw ROI press `LB` on the mouse and drag the cursor until you are satisfied
with the result.
   1. Drawing mode turns off automatically after the ROI is drawn.
   2. To clear the ROI press `RB` or `Del` for Linus users.
   3. To draw another ROI clear it using previous list item ⬆️ and
   repeat everything listed above in this section. 

#### 🟢 Navigation
 - To switch to the previous frame press `p`
 - To switch to the next frame press `n`

#### 🔵 Finish & Save
 Press `q` or `Esc` to finish and to save the progress as a \<filename>_imp.json 
### Usage
```sh
$ ./orbAnalysis.py -h

usage: orbAnalysis.py [-h] [-vid VIDEO] [-a ANNOTATIONS]
                      [-hor HORIZONTAL | -ver VERTICAL] [-wsize WSIZE]

Document Taxonomy Builder.

optional arguments:
  -h, --help            show this help message and exit
  -vid VIDEO, --video VIDEO
                        path to video (default: test-
                        parameters/Cflo_troph_count_masked_6-00_6-31.mp4)
  -a ANNOTATIONS, --annotations ANNOTATIONS
                        path to the MAL annotations (default: test-parameters/
                        Cflo_troph_count_masked_6-00_6-31_MAL_withId.json)
  -hor HORIZONTAL, --horizontal HORIZONTAL
                        type of images' stack (default: None)
  -ver VERTICAL, --vertical VERTICAL
                        type of images' stack (default: None)
  -wsize WSIZE          Your screen parameters WxH (default: 1600x1200)
```
### Examples
To run the code on test parameters simply call
```sh
./orbAnalysis.py
```
To run the code on <annotation.json> on <video.mp4> and stick it vertically. 
```sh
./orbAnalysis.py -vid video.mp4 -a annotation.json -ver true
```
## 🛡️ lbxTorch

### Description
Evaluation and conversion script for counting annotated objects on one label imported from Labelbox
and converting latter annotations into YOLOv5 format.