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
### Description
The script displays two successive consecutive frames, where annotated objects are surrounded by bounding boxes.
Annotations should follow the [Labelbox style](https://docs.labelbox.com/reference/bounding-box#export)
for the video annotations exporting. To simplify the ID tracking, objects with same ID are connected with the lines.
Possibilities of the script:
#### 🔴 Connecting lines
- Press `1` to turn on the mode where each line is visualised.
Line and bbox gets highlighted if the user hovers the mouse over the object.
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
   2. To clear the ROI press `RB` or `Del` for Linux users.
   3. To draw another ROI clear it using previous list item ⬆️ and
   repeat everything listed above in this section. 

#### 🟢 Navigation
 - To switch to the previous frame press `p`
 - To switch to the next frame press `n`

#### 🔵 Finish & Save
 Press `q` or `Esc` to finish and to save the progress as a \<filename>_imp.json 
### Usage
```commandline
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

### Description and Usage
Evaluation and conversion script for counting annotated objects on one label imported from Labelbox
and converting latter annotations into YOLOv5 format.

```commandline
$ ./lbxTorch.py -h

usage: lbxTorch.py [-h] -json-path FILEPATH [FILEPATH ...] [-s FRAME_SIZE] [-o OUTP_DIR] [-f FRAMES] [-k]

Document Taxonomy Builder.

optional arguments:
  -h, --help            show this help message and exit
  -json-path FILEPATH [FILEPATH ...], --filepath FILEPATH [FILEPATH ...]
                        Path for json files (default: None)
  -s FRAME_SIZE, --frame-size FRAME_SIZE
                        The size format is WxH, for example: 800x600 (default: None)
  -o OUTP_DIR, --outp-dir OUTP_DIR
                        Output directory for the label files (default: /home/valia/PycharmProjects/AntDet_YOLOv5/labels)
  -f FRAMES, --frames FRAMES
                        Range of frames (default: 1-$)
  -k, --keyframed-objects
                        True if annotations should be counted (default: False)

```

Script supports two different scenarios:
#### :purple_circle: Converting the annotations from Labelbox format to YOLOv5 format
Annotations in [Labelbox style](https://docs.labelbox.com/reference/bounding-box#export) for a video
got converted into the [YOLOv5 style](https://blog.paperspace.com/train-yolov5-custom-data/).
To use the script for this task  `-s, --frame-size` argument should be passed.
```commandline
./lbxTorch.py --json-path annotations.json -f 5-14 -s 800x600
```
#### :white_circle: Count the number of objects annotated by-hand
Annotations in [Labelbox style](https://docs.labelbox.com/reference/bounding-box#export) for a video
got counted if they were annotated by-hand. To use the script for this task `-k` argument should be passed.

## :radioactive: validAnnotations.py

### Description
Validation script for correction of the by-hand video annotation exported from
[Labelbox](https://docs.labelbox.com/reference/bounding-box#export) returns the
Antlist object, which is a list with extra methods added and saved the `new.json`
file with fixed annotations. Correction looks for flying heads (object was meant
to be the ant's head, but user accidentally switched feature ids of two objects)
to warn the user about their occurrence and cuts the ant's head if the borders
of the head go beyond the borders of the body.

### Usage
```commandline
./validAnnotations.py -h

usage: validAnnotations.py [-h] -a ANNOTATIONS [-r ROI]

Document Taxonomy Builder.

required arguments:
  -a ANNOTATIONS, --annotations ANNOTATIONS
                        Path to an annotation file (default: None)
optional arguments:
  -h, --help            show this help message and exit
  -r ROI, --roi ROI     Path to the ROI file (default: None)
```
## :diving_mask: videoMask.py

### Description
Masking the trophallaxis events for ant videos.

### Usage
```commandline
./videoMask.py -h
usage: videoMask.py [-h] -v VIDPATH -r ROIS [-f FILENAME] [-c COLOR | -rand]

Document Taxonomy Builder.

required arguments:
  -v VIDPATH, --vidpath VIDPATH
                        path to video (default: None)
  -r ROIS, --rois ROIS  LEFT,TOP,WIDTH,HEIGHT[;SHAPE=rect][^FRAME_START=1][!FRAME_FINISH=LAST_FRAME]] (default: [])

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        name for a processed video (default: None)
  -c COLOR, --color COLOR
                        color written as a word like pink, aqua, etc. (default: None)
  -rand                 True if background needs to be randomly colored (default: False)

```



