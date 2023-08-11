# CS725: Homework 1: Logistic regression and linear classification

![best_lowquality anim](https://github.com/ashutoshbsathe/cs725-hw/assets/22210756/9456c49b-cc33-48ff-8b4a-831b27322392)

## TAs in-charge
* Ashutosh Sathe [@ashutoshbsathe](https://github.com/ashutoshbsathe)
* Krishnakant Bhatt [@KKBhatt17](https://github.com/KKBhatt17)

## Setting up
Make sure you have a python environment setup with latest versions of dependencies as listed in [`requirements.txt`](requirements.txt). Easiest way to install all the dependencies is to run `pip install -r requirements.txt`. If you have issues regarding the environment setup, contact the TAs at the earliest.

## Instructions
* Implement both the models (logistic regression and linear classifier) in [`model.py`](model.py). You should ideally modify only the functions in this file. If you think you need to modify anything else, contact the TAs.
* To train your models, use the following commands:

For logistic regression:
```
python train.py --dataset binary --model logistic_regression --num_epochs <num_epochs> --learning_rate <learning_rate> --momentum <momentum>
```

For linear classifier:
```
python train.py --dataset iris --model linear_classifier --num_epochs <num_epochs> --learning_rate <learning_rate> --momentum <momentum>
```

The default values for each of these parameters are available in [`args.py`](args.py)

## Submission
Once you are done with both the tasks, copy weights corresponding to your best models for each dataset into [`submission/`](submission/) directory. Also copy the completed version (implementing both models) of [`model.py`](model.py) and your observation report (with filename `report.pdf`) into the same directory. Overall, make sure your [`submission`](submission/) folder looks as below. This is crucial since the assignment will be autograded:
```
submission/
    model.py
    best_binary.weights.npy
    best_iris.weights.npy
    report.pdf
```
You can get a hint of the accuracy and loss values that autograder will use for grading your submission by running `python evaluate_submission.py` in this directory itself. The observation report should also contain roll numbers of both students in the team.

Once you are satisfied with the submission, use `tar -cvzf <roll1>_<roll2>.tar.gz submission/` to create the final submission. Only the student with lower roll number in the group needs to upload this `.tar.gz` on Moodle.

## Visualizing (Optional)
Once you have implemented the logistic regression on the `binary` dataset, you can use the file [`train_with_visualization.py`](train_with_visualization.py) instead of standard [`train.py`](train.py) to create the GIF demonstrating the evolution of decision boundary of the classifier. You can find such a GIF at the top of this README. Use `python train_with_visualization.py -h` to see the list of options available to you for customizing the GIF output and training parameters. Do note that this is an optional task meant only for improving your understanding and may require significantly more computational power than just training the model. 
