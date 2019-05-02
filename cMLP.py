"""
This file includes some definations of several class to implement
the multilayer percepton model.
"""

import numpy as np
import pandas as pd
import scipy as sp
import math

#=============Class of linear scaler==================================
class cMinMaxScaler:
    """
    Linearly transform each features in "iArray" individually into the range
    between zero and one.
    """
    #----------------------------------------
    def __init__(self):
        """
        Initialize the class.
        """
        self.max = None
        self.min = None

    #----------------------------------------
    def fit(self, iArray):
        """
        Get the minimums and maximums to be used for later scaling.
        """
        self.max = iArray.max(axis=0)
        self.min = iArray.min(axis=0)

    #----------------------------------------
    def transform(self, iArray):
        """
        Transform each columns in "iArray" into the range one and zero.
        """
        return (iArray - self.min)/(self.max-self.min)

    #----------------------------------------
    def fit_transform(self, iArray):
        """
        Fit to "iArray" and then transform it.
        """
        self.fit(iArray)
        return self.transform(iArray)

#=============Activation  Functions==================================
# 1-----------. ReLU
def fRelu(iArray, Derivative=False):
    """
    If "Derivative==False", return max("iArray", 0).
    If "Derivative==True", return the matrix of the element-wise derivative of
    ReLu with respect to "iArray"
    """
    if Derivative == True:
       return np.apply_along_axis(np.diag, -1, (iArray>0)*1)
    else:
        return np.maximum(iArray, 0)

#------------2. softmax
## 2.1 derivative of softmax for 1-d array
def fDevSoftmax(iArray):
    """
    Return 2-D np.array of the derivative of softmax(iArray) with respect to each element in 1-D "iArray".
    -"iArray": 1-D np.array
    """
    yArray = sp.special.softmax(iArray,-1)
    return np.diag(yArray) - np.matmul(yArray.reshape((-1,1)), yArray.reshape((1,-1)))

## 2.1 activation function, softmax
def fSoftmax(iArray, Derivative=False):
    """
    If "Derivative==False", applying "softmax" to the last dimension of "iArray" and return the result.
    If "Derivative==True", applying "fDevSoftmax" to the last dimension of "iArray" and return the result.
    """
    if Derivative == True:
        return np.apply_along_axis(fDevSoftmax, -1, iArray)
    else:
        return sp.special.softmax(iArray, axis=-1)



#=============Dens  Class==================================
class cDense:
    """
    class of one fullly connected layer.

    """
    #----------------------------------------
    def __init__(self, input_dim, out_dim, fActivation=None, dropout=None, l2=None, bias=True, firstLayer=False):
        """
        Initialize one fully connected layer.
        -"input_dim": the input dimension.
        -"out_dim": the output dimension.
        -"fActivation=None": the activation function.
        -"dropout=None": the rate of the neurons to be dropped out. It should be with [0,1) or None.
          For default value "None", no dropout is used.
        -"l2=None": the L2 regulariztion parameter. For default value, "None", no regulariztion is used.
        -“bias=Ture”: For True, bias is added and for false, bias is not used.
        -"firtLayer=False": If this is the first layer, set it to be True. Otherwise, set it to be Fasle.
        """
        self.bias = bias
        # if bias, add 1 to input_dim
        self.input_dim = input_dim+1 if self.bias==True else input_dim
        self.out_dim = out_dim
        self.fActivation = fActivation
        # if dropout is given, get the number of units in this layer to be dropped in training.
        self.dropout = None if dropout is None else int(out_dim*dropout)
        self.l2 = l2
        # If "bias==True", this array would include both weights and bias (the last row).
        # Otherwise, only weights are included.
        self.weight = np.array(np.random.uniform(-1, 1, (self.input_dim, self.out_dim)), dtype=np.float64)

        # firstLayer: True for first layer; False for other layera
        self.firstLayer = firstLayer

        # parameters for adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.beta1t = 1.0
        self.beta2t = 1.0
        self.m = 0.0
        self.v = 0.0

    #----------------------------------------
    def fForward(self, input_data, flag_train=True):
        """
        Do forward propagation for training and prediction.
        """
        self.input = np.hstack((input_data, np.ones((input_data.shape[0],1)))) if self.bias == True else input_data
        self.Z = np.matmul(self.input, self.weight) # the result before activated.

        # drop out
        if self.dropout is not None:
            # train
            if  flag_train==True:
                # the list of columns to be dropped.
                self.dropList = np.random.choice(self.weight.shape[1], self.dropout, replace=False)
                self.Z[:, self.dropList] = 0
            # prediction
            else:
                self.Z = self.Z*(1-self.dropout/self.out_dim)

        self.out = self.Z if self.fActivation is None else self.fActivation(self.Z)

    #----------------------------------------
    def fAdam(self, g):
        """
        implement the adam optimization algorithm.
        -"g": the gradient.
        """
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.v = self.beta2*self.v + (1-self.beta2)*np.square(g)
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        mhat = self.m/(1-self.beta1t)
        vhat = self.v/(1-self.beta2t)

        # return np.sqrt(1-self.beta2t)/(1-self.beta1t) * self.m/(np.sqrt(self.v)+self.epsilon)
        return mhat/(np.sqrt(vhat)+self.epsilon)


    #----------------------------------------
    def fBackward(self, partialLoss, learning_rate=0.001, adam=True):
        """
        Do backward propagation in training to get the derivative of loss with respect to input and update weights and biases.
        If "adam==True", the adam opimizer is used to update the parameter.
        """
        # activation
        if self.fActivation is not None:

            # reshape "partialLoss" to (N, 1, M)
            tempArray = partialLoss.reshape((partialLoss.shape[0],1,partialLoss.shape[1]))

            # "self.Q" is the derivative of the activation function with repect to the input row-wise,
            # "Q[N,i,j] = d(Y[N,i])/d(Z[N, j])" with Y denoting the out and N denoting the row index.
            self.Q = self.fActivation(self.Z, True)

            # calculate LQ and reshape it back to (N, M)
            # "self.LQ" is the row-wise matrix multiplication between "partialLoss" and "self.Q",
            # "LQ[N,j] = partialLoss[N,:]*Q[N,:,j]"
            self.LQ = np.matmul(tempArray, self.Q).reshape((partialLoss.shape[0], partialLoss.shape[1]))
        # no activation
        else:
            self.LQ = partialLoss

        # get the partialLoss except the first layer
        if self.firstLayer == True:
            self.partialLoss = 0
        else:
            self.partialLoss = np.matmul(self.LQ, self.weight.T)[:,:-1]

        # gradient of loss with respect to weights
        deltaW = np.matmul(self.input.T, self.LQ)/self.input.shape[0]
        if self.l2 is not None:
            deltaW = deltaW + self.l2*self.weight
        # adam
        if adam == True:
            deltaW = self.fAdam(deltaW)

        self.weight -= (learning_rate*deltaW)


#=========================class of multilayer percepton===============
class cMLP_Model:
    """
    Class to construct MLP model
    """
    #----------------------------------------
    def __init__(self, listLayers, list_fActivation, listDrop, listL2, listBias):
        """
        Initialize the calss.
        -"listLayers": the list of numbers of units in each layers. The first and last elements are the dimensions of input and out respectively.
        -"list_fActivation": the list of the activation functions used by the corresponding layers.
           Its length should be 1 less than that of "listLayers".
        -"listDrop": the dropout rates used by each layer. Its length should be 1 less than that of "listLayers".
        -"listL2": the l2 regularization used by each layer. Its length should be 1 less than that of "listLayers".
        -"listBias": the list of booleans to denoting whether the bias is used in each layer.  Its length should be 1 less than that of "listLayers".
        """
        self.listLayers = []
        # stack layers
        i = 0
        self.listLayers.append(cDense(listLayers[i], listLayers[i+1], fActivation=list_fActivation[i],
                                      dropout=listDrop[i], l2=listL2[i], bias=listBias[i], firstLayer=True))
        for i in range(1, len(listLayers)-1):
            self.listLayers.append(cDense(listLayers[i], listLayers[i+1], fActivation=list_fActivation[i], dropout=listDrop[i], l2=listL2[i], bias=listBias[i]))

    #----------------------------------------
    def fForward(self, input_data, flag_train=True):
        """
        Do forward propagation for training and prediction.
        -"input_data": the input data of features
        -"flag_train=True": True for training and False for prediction.
        """
        for i in range(len(self.listLayers)):
            self.listLayers[i].fForward(input_data, flag_train)
            input_data = self.listLayers[i].out

    #----------------------------------------
    def fBackward(self, partialLoss, learning_rate=0.001, adam=True):
        """
        Do backward propagation in training to update weights and biases.
        -"partialLoss": the derivative of loss with respect to the output of MLP.
        -"learning_rate=0.001": learning rate.
        -"adam": True to use the adam optimization algorithm to update the parameters.
        """
        # do backward propagation one layer by one layer.
        for i in range(len(self.listLayers)):
            self.listLayers[-i-1].fBackward(partialLoss, learning_rate, adam)
            partialLoss = self.listLayers[-i-1].partialLoss

    #----------------------------------------
    def fPre(self, input_data):
        """
        Do prediction on input_data
        """
        self.fForward(input_data, flag_train=False)
        return self.listLayers[-1].out

    #----------------------------------------
    def fLoss(self, prediction, label):
        """
        Calculate the cross entropy loss betwee "prediction" and "label".
        """
        loss = -np.matmul((np.log(prediction)).reshape((1,-1)), label.reshape((-1,1)))/label.shape[0]
        return loss[0,0]

    #----------------------------------------
    def fdLoss(self, prediction, label):
        """
        Calculate the derivative of loss with respect to the prediction
        """
        prediction = np.apply_along_axis(np.diag, -1, 1/prediction)
        # reshape "label" to (N, 1, M)
        label = label.reshape((label.shape[0], 1, label.shape[1]))
        # do the row-wise matrix multiplication between prediction and label
        return -np.matmul(label, prediction).reshape((label.shape[0], label.shape[-1]))


    #----------------------------------------
    def fAccuracy(self, prediction, label):

        """
        Calculate the accuracy rate
        """
        return np.sum(np.equal(np.argmax(prediction, -1), np.argmax(label, -1)))/label.shape[0]


    #----------------------------------------
    def fTrain(self, input_data, label, epochs, batch_size, learning_rate, adam, val_data=None, verbose=True):
        """
        Train model.
        -"input_data": input data.
        -"label": the target label.
        _"epochs": the epochs.
        -"batch_size": the batch size.
        -"learning_rate": the learning rate.
        -"adam": True to use the adam optimizer. False to use the stochastic gradient optimizer.
        -"val_data=None": If it is given, it should be a list or tuple, like [val_data, label_val].
          Then the loss and accuracy of the validation data would be printed after each traing batch.
        """
        indices = np.arange(label.shape[0])
        history = {"train_loss":[], "train_acc":[]}
        if val_data is not None:
            history["val_loss"]=[]
            history["val_acc"]=[]
        # loop over each epoch
        for e in range(epochs):
            print("Epoch {}==================================: ".format(e+1))
            np.random.shuffle(indices) #shuffle the indices
            # loop over each minibatch
            for i in range(math.ceil(label.shape[0]/batch_size)):

                iIndex = i*batch_size
                input_batch = input_data[indices[iIndex: iIndex+batch_size]]
                label_batch = label[indices[iIndex: iIndex+batch_size]]

                # forward to calculate the prediction
                self.fForward(input_batch, flag_train=True)
                # get partialLoss
                partialLoss = self.fdLoss(self.listLayers[-1].out, label_batch)
                # backward to update weights
                self.fBackward(partialLoss, learning_rate, adam)

                #------------
                # print the loss and accuracy on train batch
                train_loss, train_acc = self.fEvaluate(input_batch, label_batch, "Train", verbose=verbose)
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                # print the loss and accuracy on validation data
                if val_data is not None:
                    val_loss, val_acc = self.fEvaluate(*val_data, "val", verbose=verbose)
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)
                    # print(f"Val_loss: {val_loss}; val_accuracy: {val_acc}")

        return history

    #----------------------------------------
    def fEvaluate(self, input_data, label, key=None, verbose=True):
        """
        Evluate the
        """
        pre = self.fPre(input_data)
        loss = self.fLoss(pre, label)
        acc = self.fAccuracy(pre, label)
        if verbose==True:
            if key:
                print(f"{key}_Loss: {loss}; {key}_Acc: {acc}.")
            else:
                print(f"Loss: {loss}; Acc: {acc}.")
        return loss, acc
