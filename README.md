# Amazon Warehouse bin size verification using YOLO - Deep Learning

## overview
this project implements a **yolov5 deep learning model** to detect and count items in **amazon warehouse bins**, enhancing automation accuracy and reducing operational errors. images from the **amazon bin image dataset** were preprocessed, annotated, and used to train and validate the model on google cloud with gpu support.

## key features
- **data preprocessing**: cleaning and enhancing images (clahe, log, bilateral filtering) for better clarity.  
- **annotation**: manual bounding boxes created in yolo format using labelimg.  
- **model training**: yolov5 training on google cloud vertex ai with gpu acceleration.  
- **evaluation**: performance measured with accuracy, recall, f1-score, mAP50, and confusion matrix.  
- **automation-ready**: optimized to support real-time detection and counting in warehouse workflows.  

## repository structure
```

DataCleaning\_Steps.ipynb             # data cleaning and preprocessing scripts
Data\_analysis\_images.ipynb           # exploratory analysis and visualization
GwarProc\_preprocessing.ipynb         # additional preprocessing for data pipeline
YOLO\_model\_newboundingbox.ipynb      # notebook for updated YOLO training workflow
final\_code\_YOLO.ipynb                # final YOLOv5 training and inference code
increasing\_images.ipynb              # notebook for dataset augmentation
clean\_dataset.xlsx                   # processed dataset for training/validation
LICENSE                              # mit license
Report.pdf                           # full project report with methods and results

````

## environment & tools
- **languages & libraries:** python, pandas, numpy, scikit-learn, matplotlib  
- **deep learning frameworks:** pytorch, yolov5, tensorflow  
- **cloud & storage:** google cloud vertex ai, google cloud storage  
- **annotation tools:** labelimg  

## usage
1. **clone the repository**
   ```bash
   git clone <your-repo-url>
   cd amazon-bin-yolo
````

2. **install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **train the model**

   ```bash
   python train.py --data data.yaml --cfg yolov5s.yaml --epochs 50 --batch 16
   ```

4. **run detection**

   ```bash
   python detect.py --source data/images/test --weights runs/train/weights/best.pt
   ```

## results

* **accuracy:** \~41%
* **recall:** \~79.7%
* **mAP50:** \~55.4%
* **f1-score:** \~56%

these results demonstrate that the yolov5 model effectively supports **automation-ready object detection** in amazon warehouse settings.

## license

this project is licensed under the [MIT License](./LICENSE).


