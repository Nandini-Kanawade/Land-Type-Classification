# Land Cover Classification using Convolutional Neural Networks (CNN)

## Overview
This project aims to develop and evaluate a convolutional neural network (CNN) model for land cover classification using satellite imagery. The project utilizes the EuroSAT dataset, a widely used benchmark dataset for land cover classification tasks.

## Project Image
![Project Image](https://github.com/Nandini-Kanawade/Land-Type-Classification/blob/847b638f896ee75bc2aaed829c4c2f6a0619484f/WebSite_Image)

### Dataset
The EuroSAT dataset consists of satellite images covering 13 spectral bands at a spatial resolution of 10 meters per pixel. It contains 27,000 labeled images covering 10 different land cover classes including Annual Crop, Pasture, Forest, Highway, Residential, and more.

### Methodology
The methodology involves preprocessing the dataset, utilizing the VGG16 model architecture for feature extraction, fine-tuning the model, and evaluating its performance.

1. **Preprocessing**: Includes steps like downloading, extracting, reading, filtering, balancing, splitting, and encoding the data to prepare it for training the model effectively.
2. **Model Architecture**: Utilizes the VGG16 model, a convolutional neural network architecture known for its effectiveness in image classification tasks.
3. **Training**: Involves loading a pre-trained VGG16 model, freezing its weights, adding custom layers for classification, compiling, and training the model.
4. **Evaluation**: Evaluates the model's performance on test data before and after fine-tuning.

### Evaluation
The model's performance is evaluated based on accuracy metrics before and after fine-tuning. The accuracy improved from 64% to 82% after fine-tuning.

## References
- EuroSAT dataset: [Link](https://github.com/phelber/EuroSAT)

