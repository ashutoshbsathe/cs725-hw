import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d+1, self.num_classes))
    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        pass

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        pass

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        pass

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        pass

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        pass

class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3 # 3 classes
        self.d = 4 # 4 dimensional features
        self.weights = np.zeros((self.d+1, self.num_classes))
    
    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        pass

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        pass

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        pass

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        pass

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        pass
