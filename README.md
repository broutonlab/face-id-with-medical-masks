# Broutonlab face recognition with medical masks
identification systems which is we use for unlocking our devices have struggled with medical masks appearing on human faces.

## Colab Demo
The whole pipeline code with detailed description provided in google [colab notebook](https://colab.research.google.com/gist/poltavski/23f242d5e50de9ddd1ade0d7baf8fd83/face_recognition_with_masks.ipynb).

## Solution
We will show and build system with the most modern state-of-the-art methods  possible to solve the task of face recognition with medical masks. 
In order to do that, we will make such augmentations that transform our initial training dataset into persons wearing medical masks.

![Trump](https://cdn-images-1.medium.com/max/1200/1*qFYQo4nqwc-wE_EseswvqA.png)

## Process of facial keypoints extraction
![Keypoints](https://cdn-images-1.medium.com/max/1200/1*-W7gdhRji16sBgqERt-S5Q.png)

## Triangulation process
![Triangulation](https://cdn-images-1.medium.com/max/1200/1*-KyFG7mHQnh9vdqkkpxyDA.png)

## Medical mask matching
![Mask](https://cdn-images-1.medium.com/max/1200/1*sWTq9BBCbKIea7tNgUh7Vg.png)

## Situation with the face rotation

Proposed solution also handles the situation with the face rotation, as medical masks database is stored in json with the calculated parameter of rotation, which allow us to match images with face rotation for only with those masks that are falling in concrete interval of rotation for given face.

![Rotation](https://cdn-images-1.medium.com/max/1200/1*p0wp1UTrM5Wj3RsgDpZ9vg.png)

## ArcFace

Process of training a DCNN for face recognition supervised by the ArcFace loss

![ArcFace](https://cdn-images-1.medium.com/max/2560/1*T3wkuUKIqMunwfOoi5_kGg.png)
