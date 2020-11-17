# Geometric properties of adversarial images

**Bogdan Ivanyuk-Skulskiy (NaUKMA), Galyna Kriukova (NaUKMA), Andrii Dmytryshyn (Örebro University)**

**Paper:** [https://ieeexplore.ieee.org/abstract/document/9204251](https://ieeexplore.ieee.org/abstract/document/9204251)

**DSMP 2020**

## Abstract
Machine learning models are now widely used in a variety of tasks. However, they are vulnerable to adversarial perturbations. These are slight, intentionally worst-case, modifications to input that change the model’s prediction with high confidence, without causing a human eye to spot a difference from real samples. The detection of adversarial samples is an open problem. In this work, we explore a novel method towards adversarial image detection with linear algebra approach. This method is built on a comparison of distances to the centroids for a given point and its neighbours. The method of adversarial examples detection is explained theoretically, and the numerical experiments are done to illustrate the approach.

## About the paper

## Code
To test the algorithm on other data first train autoencoder model and save model architercture in `models/autoencoder.py` and model weights in `data/model_weights/` folder. 

Afterwards run the following code:

`python run_experiment.py --data_path_real=data/train_data/real_data --data_path_generated=data/train_data/adv_data --autoencoder_weights_path=data/model_weights/my_autoencoder_full.pth --number_of_samples=1000`


### Citation
```
@INPROCEEDINGS{9204251,
  author={B. {Ivanyuk-Skulskiy} and G. {Kriukova} and A. {Dmytryshyn}},
  booktitle={2020 IEEE Third International Conference on Data Stream Mining   Processing (DSMP)}, 
  title={Geometric Properties of Adversarial Images}, 
  year={2020}
  }
```
