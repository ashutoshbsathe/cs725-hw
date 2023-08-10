# CS725: Homework 1: Logistic regression and linear classification

## TAs in-charge
* Ashutosh Sathe [@ashutoshbsathe](https://github.com/ashutoshbsathe)
* Krishnakant Bhatt [@KKBhatt17](https://github.com/KKBhatt17)

## Setting up
Make sure you have a python environment setup with latest versions of dependencies as listed in [`requirements.txt`](requirements.txt). If the latest versions don't work for you or you are unsure about the versions, contact the TAs.

## Instructions
* Implement both the models (logistic regression and linear classifier) in [`model.py`](model.py)
* To train your models, use the following commands:

For logistic regression:
```
python train.py --dataset binary --model logistic_regression --num_epochs <num_epochs> --learning_rate <learning_rate> --momentum <momentum>
```

For linear classifier:
```
python train.py --dataset digits --model linear_classifier --num_epochs <num_epochs> --learning_rate <learning_rate> --momentum <momentum>
```
