# Tahaluf-Deep-Learning-Engineer-Task


# Clothing Article Classifier

This repository contains the code and resources for a clothing article classifier using machine learning techniques. The classifier is designed to identify and categorize different types of clothing items from high-resolution images.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Handling](#data-handling)
3. [Model Architecture and Approach](#model-architecture-and-approach)
4. [Metrics](#metrics)
5. [Results and Discussion](#results-and-discussion)
6. [Computational Complexity](#computational-complexity)
7. [Conclusion](#conclusion)

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

## Computational Complexity

We calculated the number of Floating Point Operations (FLOPS) and Multiply-accumulate operations (MACCs) per layer for the main convolutional and fully connected layers in the three models: Custom CNN and ResNet50. The results are presented in the table below the most expensive layers:

| Model       | Layer Name      | FLOPSx 10^6     | MACCsx 10^6 |
|-------------|-----------------|-----------------|------------ |
| Custom CNN  | Conv2           | 9,912.32        | 4,956.16    | 
| ResNet50    | ResBlock1_Conv1 | 248.84          | 124.42      | 


**Total FLOPS and MACCs for each Model:**

| Model       | Total FLOPSx 10^6 | Total MACCsx 10^6 |
|-------------|-------------------|-------------------|
| Custom CNN  | 29,891.94         | 14,945.97         | 
| ResNet50    | 7,735.46          | 3,867.73          |
| ConvNext-T  | 4,500.46          | 2,250.23          |

**Note: The Flops of ConvNext-T couldn't get it from the calculations because the tensorflowHub model is colsed, but in future i will try to revert it to its archiecturte. So the Flops in the table is from its benchmark**

Table with the most expensive layer in each model, along with the time it takes to compute.


### Decreasing FLOPS and MACCs

To decrease the FLOPS and MACCs, we can apply various techniques, such as:

1. Decreasing the number of filters in convolutional layers: By reducing the number of filters, we can reduce the number of operations performed in each layer, thereby reducing FLOPS and MACCs.

2. Using depth-wise separable convolutions: These convolutions split the standard convolution operation into two separate operations: a depth-wise convolution followed by a point-wise convolution. This can reduce the computational complexity significantly.

3. Applying network pruning: By removing less important connections or weights from the network, we can reduce the complexity of the model without sacrificing much performance.

4. Reducing the input size: By reducing the spatial dimensions of the input image, we can decrease the number of operations performed in each layer.

### Most Computationally Expensive Layers

In the table above, the most computationally expensive layers are typically the convolutional layers with the highest number of filters and the largest filter sizes. In the case of ResNet50, the most expensive layers might be the ones within the residual blocks.

By focusing on optimizing these layers, we can significantly decrease the overall computational complexity of the models.

## Conclusion

In this project, we successfully built a clothing article classifier using machine learning techniques. The classifier achieved satisfactory performance on the imbalanced dataset using the ConvNeXt tiny feature extraction model from TensorFlow Hub with 5-fold cross-validation. We also compared different approaches and demonstrated the advantages of using pre-trained models for feature extraction in handling imbalanced datasets and improving classification performance.