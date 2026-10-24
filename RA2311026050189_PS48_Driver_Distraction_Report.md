# Driver Distraction Detection Using Deep Learning
## Problem Statement PS-48 — Project Report

**Student:** Hanumanth M  
**Register Number:** RA2311026050189  
**Institution:** SRM Institute of Science and Technology  
**Department:** AI/ML, School of Computing  
**Course:** Deep Learning Techniques (21CSC401J)  
**Faculty:** Dr. M. Thurai Pandian

---

## Abstract

This project addresses driver distraction detection as a ten-class image-classification problem using dashboard-camera images. The supplied report describes 22,424 images from 26 drivers across behaviours c0–c9. A transfer-learning model uses a frozen pretrained MobileNetV2 backbone with global average pooling, dropout, and a 10-unit classification head. Notebook results report 67.84% validation accuracy, validation loss 0.9307, and macro F1-score 0.67. The substantial gap between training and validation accuracy indicates overfitting; the model is a prototype and is not suitable as a stand-alone safety-critical system.

## 1. Problem and Objectives

Given an image of a driver, predict one of ten behaviours: safe driving, texting with either hand, talking on a phone with either hand, operating the radio, drinking, reaching behind, hair/makeup, or talking to a passenger.

Objectives:
- Inspect dataset metadata and verify all ten class folders.
- Build a ten-class classifier using MobileNetV2 transfer learning.
- Train with early stopping and monitor training/validation metrics.
- Evaluate per-class precision, recall, F1-score and confusion patterns.
- Discuss limitations and possible improvements for generalisation to unseen drivers.

## 2. Dataset and Preprocessing

The metadata file `driver_imgs_list.csv` contains 22,424 rows and three columns: driver identifier (`subject`), class (`classname`) and filename (`img`). The report records 26 unique drivers.

| Class | Behaviour | Images |
|---|---|---:|
| c0 | Safe driving | 2,489 |
| c1 | Texting, right hand | 2,267 |
| c2 | Talking on phone, right hand | 2,317 |
| c3 | Texting, left hand | 2,346 |
| c4 | Talking on phone, left hand | 2,326 |
| c5 | Operating the radio | 2,312 |
| c6 | Drinking | 2,325 |
| c7 | Reaching behind | 2,002 |
| c8 | Hair and makeup | 1,911 |
| c9 | Talking to a passenger | 2,129 |
| **Total** | | **22,424** |

The notebook preparation recorded 17,446 training images and 4,978 validation images in batches of 32. The report notes that no independent test set is shown; the validation set was used for early stopping and final reported metrics. Driver-wise separation should be checked carefully when claiming performance on previously unseen drivers.

## 3. Methodology

The model uses the pretrained `mobilenetv2_1.00_224` feature extractor with 224 × 224 RGB inputs. The backbone is frozen, followed by GlobalAveragePooling2D, dropout, and a dense output layer with ten classes. The model summary reports 2,270,794 total parameters, of which 12,810 are trainable and 2,257,984 are non-trainable.

| Parameter | Value |
|---|---|
| Backbone | MobileNetV2, frozen |
| Classification head | Global average pooling, dropout, Dense(10) |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Sparse categorical cross-entropy |
| Batch size | 32 |
| Maximum epochs | 8 in first run; early stopping ended at epoch 5 |
| Early stopping | Monitor validation loss; patience 2; restore best weights |
| Training time noted in report | First run approx. 233 seconds total |

The screenshots did not confirm every preprocessing setting, augmentation configuration, or the exact starting state of the shorter plotted run; these details should not be inferred beyond what the supplied report states.

## 4. Results

The supplied notebook/report records the following validation performance:

| Metric | Result |
|---|---:|
| Validation images | 4,978 |
| Validation accuracy | 67.84% |
| Validation loss | 0.9307 |
| Macro precision | 0.70 |
| Macro recall | 0.67 |
| Macro F1-score | 0.67 |
| Weighted F1-score | 0.68 |

### Per-class validation metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| c0 Safe driving | 0.55 | 0.70 | 0.62 | 564 |
| c1 Texting, right hand | 0.81 | 0.58 | 0.68 | 474 |
| c2 Phone, right hand | 0.56 | 0.65 | 0.60 | 501 |
| c3 Texting, left hand | 0.81 | 0.73 | 0.77 | 536 |
| c4 Phone, left hand | 0.79 | 0.59 | 0.67 | 531 |
| c5 Operating the radio | 0.88 | 0.85 | 0.86 | 526 |
| c6 Drinking | 0.85 | 0.74 | 0.79 | 530 |
| c7 Reaching behind | 0.85 | 0.88 | 0.86 | 480 |
| c8 Hair and makeup | 0.37 | 0.70 | 0.49 | 400 |
| c9 Talking to a passenger | 0.54 | 0.32 | 0.40 | 436 |

A single prediction example in the report is class c0, correctly predicted as c0 with 78.59% confidence. This single example is illustrative and does not represent overall accuracy.

## 5. Analysis and Discussion

The strongest classes by F1-score are c5 (operating the radio) and c7 (reaching behind), at 0.86 each. Class c9 (talking to a passenger) is weakest at 0.40, followed by c8 (hair and makeup) at 0.49. The report identifies c8 as an attractor class: many images from other behaviours are predicted as hair/makeup, resulting in low precision (0.37). The largest confusion reported is c2 (phone, right hand) predicted as c8 for 147 of 501 images. The report also notes c9 predicted as safe driving in 115 of 436 images, which is important because distracted behaviour may be incorrectly labelled as safe. Training accuracy exceeds 0.9 while validation accuracy remains around 0.64–0.68, indicating a clear generalisation gap.

## 6. Real-World Relevance, Limitations and Ethics

Potential applications include driver-monitoring research, fleet safety review, driver training, and safer cabin-interface design. These are potential applications only; this project implements offline image classification and evaluation, not a deployed warning system.

Limitations include moderate validation performance, overfitting, absence of a separate independent test set, a frozen feature extractor, no reported augmentation or class weighting, data from a limited driver/camera setting, and the use of single images without temporal context. Driver images are sensitive: any real deployment should use appropriate notice, consent, access controls, and privacy-preserving data handling. A model with these error rates must not be used as the sole basis for safety-critical decisions.

Recommended future work:
- Fine-tune upper MobileNetV2 layers with a low learning rate.
- Test suitable augmentation while carefully preserving left/right class labels.
- Hold out drivers for an independent test set and report cross-driver results.
- Compare alternative backbones and inspect a gallery of misclassified images.
- Evaluate multiple video frames and test across lighting, cameras and vehicle interiors.

## 7. Conclusion

The project implements a ten-class distracted-driver image classifier using MobileNetV2 transfer learning. The supplied outputs report 67.84% validation accuracy, 0.9307 validation loss and 0.67 macro F1-score. Stronger recognition of radio operation and reaching behind contrasts with weaker recognition of talking to a passenger and hair/makeup. Further generalisation testing, fine-tuning and error analysis are needed before considering any real-world safety use.

## References

1. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” CVPR, 2018, pp. 4510–4520.
2. State Farm, “State Farm Distracted Driver Detection,” Kaggle competition. https://www.kaggle.com/c/state-farm-distracted-driver-detection
3. Keras, “Keras Applications: MobileNet, MobileNetV2 and MobileNetV3.” https://keras.io/api/applications/mobilenet/
4. TensorFlow, “Transfer learning and fine-tuning.” https://www.tensorflow.org/tutorials/images/transfer_learning
5. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” ICLR, 2015.
6. N. Srivastava et al., “Dropout: A simple way to prevent neural networks from overfitting,” Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.
7. F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

---

*This Markdown report is a readable text version prepared from the supplied PDF/DOCX report. It does not embed the original report's screenshots and figures. Reported metrics are reproduced from the supplied report; they are not independently re-run here.*
