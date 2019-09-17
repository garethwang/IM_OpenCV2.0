# Image Matching Pipeline

A c++ implementation of image matching.

## Description

The major stages of image matching consist of generating features, matching descriptors and pruning matches. The corresponding methods in each stage are available here:

- Generating features 
  - SIFT
  - SURF
  - ORB
  - AKAZE
  - ROOTSIFT
  - HALFSIFT

 - Matching descriptors
   - BruteForce
   - FlannBased
 - Pruning matches
   - Ratio test
   - GMS
   - LPM

## Requirement

- OpenCV

## How to use

Read **DEMO.md**.

## Author

Gareth Wang  

- email: gareth.wang@hotmail.com