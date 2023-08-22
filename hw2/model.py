import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitGenericClassifier(pl.LightningModule):
    """
    General purpose classification model in PyTorch Lightning.
    The 2 models for the 2 respective datasets are inherited from this class.

    The 2 inherited classes define the model along with the choice of the optimizer.
    Rest of the code which is responsible for setting up training is common to both.
    """
    def __init__(self, lr=0):
        super().__init__()
        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss()
        self.model = nn.Sequential() # modify this in individual model class

    def training_step(self, batch, batch_idx=0):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the call. It is your responsibility to implement this function in both
        models. If you are not preprocessing the data, either don't call it at all or add a dummy 
        function as 
        ```
        def transform_input(self, batch):
            return batch
        ```
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much training data has been used within an epoch.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` with correct loss value and gradient. If you are using PyTorch 
        operations, the gradient should not be destroyed. If your model is not improving or if 
        the loss becomes NaN, check this loss computation very carefully and make sure it preserves
        gradient for the autograd engine.
          PyTorch Lightning will automatically take the `loss` and run `loss.backward()` to compute 
        gradient and update weights by calling `optim.step()`. You just need to return the `loss`
        appropriately. We log these values every step so that it is easier to compare various runs.
        """
        loss = ...
        acc = ...
        self.log('train_loss', loss.item())
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx=0):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the actual implementation.
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much validation data has been used during evaluation.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` or a scalar with loss value. Gradient is not required here.
        `acc`: A `torch.Tensor` or a scalar with accuracy value between 0 to 1.
          These values will be useful for you to assess overfitting and help you determine which model
        to submit on the leaderboard and in the final submission.
        """
        loss = ...
        acc = ...
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        return {
            'valid_loss': loss,
            'valid_acc': acc,
        }

    def test_step(self, batch):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the actual implementation. 
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much validation data has been used during evaluation.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` or a scalar with loss value. Gradient is not required here.
        `acc`: A `torch.Tensor` or a scalar with accuracy value between 0 to 1.
          This function is very similar to `validation_step` and will be used by the autograder while
        evaluating your model. You can simply copy over the code from `validation_step` into this if 
        you wish. Just ensure that this calculation is correct.
        """
        loss = ...
        acc = ...
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {
            'test_loss': loss,
            'test_acc': loss,
        }
    
    def predict(self, x):
        """
        Arguments
        =================
        `x`: `torch.Tensor` of size (B, d) such that B = batch size and d = input feature dimensions.
          You can optinally transform this appropriately using `self.transform_input(batch)` but you 
        may need to create fake labels so that the function call stays the same. Something like this
        could work: `self.transform_input((x, torch.zeros(x.size(0)).long()))`
        
        Operation
        =================
        Classify each instance of `x` into appropriate classes.

        Returns
        =================
        `y_pred`: `torch.LongTensor` of size (B,) such that `y_pred[i]` for 0 <= i < B is the label
        predicted by the classifier for `x[i]`
        """
        y_pred = ...
        return y_pred

class LitSimpleClassifier(LitGenericClassifier):
    def __init__(self, lr=0):
        super().__init__(lr=lr)
        self.model = nn.Sequential(
            nn.Linear(2, ...), # d = 2
            ..., # build your model here using `torch.nn.*` modules
            nn.Linear(..., 4)  # num_classes = 4
        )

    def transform_input(self, batch):
        # hardcode your transform here
        return batch

    def configure_optimizers(self):
        # choose an optimizer from `torch.optim.*`
        # use `self.lr` to set the learning rate
        # other parameters (e.g. momentum) may be hardcoded here
        return torch.optim.<optim_name>(...)

class LitDigitsClassifier(LitGenericClassifier):
    def __init__(self, lr=0):
        super().__init__(lr=lr)
        self.model = nn.Sequential(
            nn.Linear(64, ...), # d = 64
            ..., # build your model here using `torch.nn.*` modules
            nn.Linear(.., 10)   # num_classes = 10
        )
    
    def transform_input(self, batch):
        # hardcode your transform here
        return batch

    def configure_optimizers(self):
        # choose an optimizer from `torch.optim.*`
        # use `self.lr` to set the learning rate
        # other parameters (e.g. momentum) may be hardcoded here
        return torch.optim.<optim_name>(...)
