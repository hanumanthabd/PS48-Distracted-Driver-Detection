# Driver Distraction Detection Using Deep Learning (PS-48)

**Course:** Deep Learning Techniques (21CSC401J)  
**Student:** Hanumanth M  
**Register Number:** RA2311026050189  
**Institution:** SRM Institute of Science and Technology  
**Model:** MobileNetV2 transfer learning  
**Problem Statement:** PS-48 — Driver Distraction Detection

## Overview

This project classifies in-car dashboard-camera images into ten driver-behaviour classes, from safe driving to common distractions such as phone use, drinking, operating the radio, reaching behind, hair/makeup, and talking to a passenger.

## Dataset

The report describes 22,424 labelled images across 10 classes and 26 drivers. The notebook preparation recorded 17,446 training images and 4,978 validation images (batch size 32). Keep the dataset itself out of this repository; follow the dataset provider's terms and link to the original source when sharing it.

## Method

- Pretrained MobileNetV2 feature extractor, frozen during initial training
- Global average pooling, dropout, and a 10-unit classification output
- Adam optimizer (learning rate 0.001)
- Sparse categorical cross-entropy loss
- Early stopping monitoring validation loss

## Reported results

According to the supplied report/notebook outputs:
- Validation accuracy: **67.84%**
- Validation loss: **0.9307**
- Macro F1-score: **0.67**

These are validation metrics, not results from an independent held-out test set. The report notes a gap between training and validation performance, indicating overfitting and room for improvement.

## Repository files

- `RA2311026050189_PS48_Driver_Distraction_Report.pdf` — project report (PDF)
- `RA2311026050189_PS48_Driver_Distraction_Report.docx` — editable report (Word)
- `README.md` — project overview

The Google Colab notebook has not been included yet. Add the executed `.ipynb` notebook here to make the code reproducible.

## Limitations and responsible use

The model's validation accuracy and class-specific errors mean it should not be used as a sole safety-critical driver-monitoring system. Real-world use would require further testing across drivers, vehicles, lighting, camera positions, and demographic variation, plus appropriate privacy and consent protections.

## Dataset reference

State Farm Distracted Driver Detection (Kaggle): https://www.kaggle.com/c/state-farm-distracted-driver-detection
