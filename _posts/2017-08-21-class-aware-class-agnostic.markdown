---
layout:     post
title:      "Class-aware v.s. Class-agnostic in object detection"
date:       2017-08-21 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Segmentation and Detection
---
> Answered by [Kien-Huynh](https://www.quora.com/profile/Kien-Huynh) in Quora


For a class-aware detector, if you feed it an image, it will return a set of bounding boxes, each box associated with the class of the object inside (i.e. dog, cat, car). It means that by the time the detector finished detecting, it knows what type of object was detected.

For class-agnostic detector, it detects a bunch of objects without knowing what class they belong to. To put it simply, they only detect “foreground” objects. Foreground is a broad term, but usually it is a set that contains all specific classes we want to find in an image, i.e. foreground = {cat, dog, car, airplane, …}. Since it doesn’t know the class of the object it detected, we call it class-agnostic.

Class-agnostic detectors are often used as a pre-processor: to produce a bunch of interesting bounding boxes that have a high chance of containing cat, dog, car, etc. Obviously, we need a specialized classifier after a class-agnostic detector to actually know what class each bounding box contains.