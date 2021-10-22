[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jejobueno/ImmoEliza-Regressions/main.svg)](https://results.pre-commit.ci/latest/github/jejobueno/ImmoEliza-Regressions/main) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<div align = "center">

![image](https://user-images.githubusercontent.com/34608190/138427104-c02b202d-f95a-4f13-bd12-d3e1f109ed3f.png)

<h3>Signature Recognition Use Case for B&D</h3>

 </div>


# Signature Recognition using YOLOv5
Machine learning model to predict prices on Belgium's real estate sales.

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)  
[Output](#Output)  
[How it works](#How-it-works)  
[Examples](#Examples)  
[Authors](#Authors)

## Description
The model predicts the prices of properties in Belgium, based on data that were gathered in a previous scraping project.
In relation with the postal code, the state of the construction, the property subtype (apartment, studio, villa, chalet, ...),
and existance of a fireplace, terrace, garden and/or fully equiped kitchen, an estimate of the asking price is made.

The accuracy of the model is 0.89, which means that there is always a possibility for outliers (less then 11 %). More importantly: in 89 %
of the cases the prediction will be within a respectable range.

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

## Output
When you run the program, you will get: 

## How it works

## Examples


## Authors
Jesús Bueno - Junior AI Developer
