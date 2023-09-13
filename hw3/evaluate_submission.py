import numpy as np
import pandas as pd
import pickle as pkl
from model import NaiveBayes

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def evaluate_model():
    print("Loading Model from model.pkl")
    try:
        loaded_model = load_model()
    except Exception as e:
        print("Caught Exception in Loading Model")
        print(e)
        return
    print("Model Successfully Loaded")
    print("Calling getParams")
    try:
        allparams = loaded_model.getParams()
    except Exception as e:
        print("Unable to call getParams()")
        print(e)
        return
    print("Following Params Loaded:")
    print("Priors:",allparams[0])
    print("Gaussian:",allparams[1])
    print("Bernoulli:",allparams[2])
    print("Laplace:",allparams[3])
    print("Exponential:",allparams[4])
    print("Multinomial:",allparams[5])
    
    ### Write code for calling predictions
    print(32*"-")
    print("Loading test dataset")
    try:
        test_dataset = pd.read_csv("./data/validation_dataset.csv",index_col=0).to_numpy() ## We will be using a hidden dataset here
        
    except Exception as e:
        print("Unable to load test.csv")
        print(e)
        return
    test_datapoints = test_dataset[:,:-1]
    test_labels = test_dataset[:, -1]
    print("Test dataset loaded successfully")
    print("Calling predict")
    try:
        predictions = loaded_model.predict(test_datapoints)
    except Exception as e:
        print("Unable to call predict")
        print(e)
        return
    print("Predictions generated successfully")
    accuracy = np.sum(predictions==test_labels)/predictions.size
    print("Accuracy on test dataset:",accuracy)

    return





def main():
    print(32*'-')
    evaluate_model()

if __name__ == '__main__':
    loaded_model = load_model() 
    main() 