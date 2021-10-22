[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jejobueno/ImmoEliza-Regressions/main.svg)](https://results.pre-commit.ci/latest/github/jejobueno/ImmoEliza-Regressions/main) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<div align = "center">

![image](https://user-images.githubusercontent.com/34608190/138427104-c02b202d-f95a-4f13-bd12-d3e1f109ed3f.png)

<h3>Signature Recognition Use Case for B&D</h3>

 </div>


# Signature Recognition using YOLOv5
Deep Learning model trained to detect signatures from diferent documents.

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)  
[Output](#Output)  
[How it works](#How-it-works)  
[Examples](#Examples)  
[Authors](#Authors)

## Description
YOLOv5 model training to detect signatures on documents, it was trained with annotated documents transformed to jpg and addapting their annotations from an .xml format to a .txt normalizing and transpolating the coordinates to the yolo format. 

## Installation
Clone the repository:
```
https://github.com/jejobueno/Signature-Recognition-BD.git
``` 
 
Install the requirements:
```
pip install -r requirements.txt
``` 

## Usage
  

To train the model you could use the following command:
```
python train.py --img 640 --batch 16 --epochs 300 --data custom_dataset.yaml --weights yolov5s.pt --cache
```   

To detect signatures from a determinated folder user the following command:
```
python detect.py --weights ./runs/train/exp/weights/best.pt --img 640 --conf 0.60 --source ./data/test
```   

This command will read all files from the folder path `/data/test` and mark all the detected signatures with their confidence score inside the `runs/detect/exp/` folder. If you don't want them just add the argument `--nosave`

## Output
When you run the program, you will get a csv with the boundry boxes coordinates and confidences called `bboxes.csv`, also another csv file marking if the images contains any signature or not calles `results_yolo.csv`

If you runned the command wwithour the argument `--nosave` you'll find also insde the folder `runs\detect\exp\` the images with the signatures detected as following image

![image](https://user-images.githubusercontent.com/34608190/138463807-7ec615ee-2b93-4e83-8291-828c151b9905.png)

This model works with a 94,4382% of accuracy detecting signatures on test set provided by the client.
## Examples
![image](https://user-images.githubusercontent.com/34608190/138463860-1aca4e38-4533-4d8f-a977-cacc8067e5cd.png)
![image](https://user-images.githubusercontent.com/34608190/138463887-f8014e0e-02fd-4929-a8a8-163c7fda6987.png)


## Authors
Jesús Bueno - Junior AI Developer
