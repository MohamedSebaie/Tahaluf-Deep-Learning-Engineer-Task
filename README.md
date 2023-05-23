# Tahaluf-Deep-Learning-Engineer-Task


# Clothing Article Classifier

This repository contains the code and resources for a clothing article classifier using machine learning techniques. The classifier is designed to identify and categorize different types of clothing items from high-resolution images.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Handling](#data-handling)
3. [Model Architecture and Approach](#model-architecture-and-approach)
4. [Metrics](#metrics)
5. [Results and Discussion](#results-and-discussion)
6. [Conclusion](#conclusion)

## Introduction

In this project, we aim to build a clothing article classifier that can accurately identify and categorize different types of clothing items from high-resolution images. This classifier can be used in various applications, such as e-commerce platforms, inventory management systems, and more.

## Data Handling

### Dataset

We used the Clothing dataset (full, high resolution) from Kaggle, which contains 5,000 images of clothes from 20 different classes. The dataset is released under CC0. Some of the items are labeled "Not sure", "Others", or "Skip" and may contain labeling errors. We removed these three classes and trained our models on the remaining 17 classes, which are imbalanced.

### Data Loading

We used Keras for data loading and preprocessing. The data was divided into training, validation, and testing sets, with a suitable split.

### Data Preprocessing

The images were resized and normalized to have zero mean and unit variance. We also applied data augmentation techniques, such as random horizontal flipping and random rotation, to increase the training dataset's diversity and improve the classifier's generalization capabilities.

## Model Architecture and Approach

We explored four different approaches:

1. Custom CNN without any pre-trained models
2. ResNet50 with 17 classes
3. ResNet50 with only the top 10 classes to handle the imbalanced dataset
4. ConvNeXt tiny feature extraction model from TensorFlow Hub

After comparing the performance and efficiency of these models, we chose the ConvNeXt tiny feature extraction model from TensorFlow Hub as our final approach, as it performed well on the imbalanced dataset. We then applied 5-fold cross-validation to the ConvNeXt tiny feature extraction model to ensure its highest accuracy and robustness.

## Metrics

We selected F1-score as our primary metric since it is suitable for imbalanced datasets and directly reflects the model's ability to correctly classify clothing items across all classes. Additionally, we monitored accuracy as a secondary metric.

## Results and Discussion

The final ConvNeXt tiny feature extraction model achieved an average F1-score of 88% using 5-fold cross-validation. This model performed well on the imbalanced dataset and demonstrated good generalization capabilities.

We compared the performance of the four approaches:

1. Custom CNN without any pre-trained models: XX% F1-score
2. ResNet50 with 17 classes: YY% F1-score
3. ResNet50 with only the top 10 classes: ZZ% F1-score
4. ConvNeXt tiny feature extraction model from TensorFlow Hub with 5-fold cross-validation: 88% average F1-score

The ConvNeXt tiny feature extraction model with 5-fold cross-validation outperformed the other approaches, making it the best choice for this problem.

## Conclusion

In this project, we successfully built a clothing article classifier using machine learning techniques. The classifier achieved satisfactory performance on the imbalanced dataset using the ConvNeXt tiny feature extraction model from TensorFlow Hub with 5-fold cross-validation. We also compared different approaches and demonstrated the advantages of using pre-trained models for feature extraction in handling imbalanced datasets and improving classification performance.
