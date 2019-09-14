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
   - GMS
   - Ratio test
   - LPM

## Requirement

- OpenCV

## Author

Gareth Wang  

- email: gareth.wang@hotmail.com