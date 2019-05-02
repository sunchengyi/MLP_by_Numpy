import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as se


from cMLP import cMLP_Model, cMinMaxScaler, fRelu, fSoftmax


def fPrepareData(files, n_classes, file_to_fit=None):
    """
    Get np.arrays from "files" and then one-hot vectorize the last column for each DF.
    If "file_to_fit==None", just return np.arrays read from "files".
    If "file_to_fit" is given, create a scaler of "cMinMaxScaler" and fit it to
    the DF from "files[file_to_fit]", and then used this scaler to
    transform all of data. Finally, return the scaled DFs and the scaler.
    -"files": the list of data files. The first column in each file should be index,
      and the last column should be target.
    -"n_classes": the number of classes for the one-hot vectorization.
    -"file_to_fit=None": the index of the file in "files" to be fitted by the scaler.
    """
    myIdentity = np.identity(n_classes)

    final_arrays=[]
    for (i, file) in enumerate(files):
        tempDF = pd.read_csv(file, index_col=0)
        tempFeatures, tempLabel = tempDF.values[:, :-1], tempDF.values[:, -1]
        tempLabel = myIdentity[np.array(tempLabel, dtype=int)] # one-hot vectorization
        final_arrays.append([tempFeatures, tempLabel])

    if file_to_fit is not None:
        scalerX = cMinMaxScaler()
        scalerX.fit(final_arrays[file_to_fit][0])
        for (i, data) in enumerate(final_arrays):
            final_arrays[i][0] = scalerX.transform(data[0])
        return scalerX, final_arrays
    else:
        return final_arrays

def main(listLayers, dropRate=None, l2=None):
    """
    Construct MLP model
    """
    list_fActivation = (len(listLayers)-2)*[fRelu] + [fSoftmax]
    listDrop = (len(listLayers)-2)*[dropRate] +  [None]
    listL2 = (len(listLayers)-1)*[None]
    listBias =  (len(listLayers)-1)*[True]
    model = cMLP_Model(listLayers, list_fActivation, listDrop, listL2, listBias)

    return model

def fEvaluate(model, datas):
    """
    datas = [[x_1, y_1], [x_2, y_2], ...]
    """
    list_loss_acc = []
    for data in datas:
        list_loss_acc.append(model.fEvaluate(data[0], data[1]))

    return list_loss_acc

def fPlot(history):
    for key in history:
        plt.plot(history[key], label=key)
    plt.legend()
    plt.show()


#======================================================
if __name__ == "__main__":
    dataFolder = "./data/"
    trainFile = dataFolder + "train_data.csv"
    testFile = dataFolder + "test_data.csv"
    valFile = dataFolder + "val_data.csv"

    files = [trainFile, valFile, testFile]
    n_classes = 3

    # prepare data, datas=[[train_x, train_y], [val_x, val_y], [test_x, test_y]]
    scalerX, datas = fPrepareData(files, n_classes, 0)

    # construct MLP model
    listLayers = [2, 12, 3]
    epochs = 20
    batch_size = 100
    learning_rate = 0.001

    # no dropout and l2
    model_a=main(listLayers, dropRate=None, l2=None)
    print("Initial loss and acc for Train, Val and Test:")
    fEvaluate(model_a, datas)
    history_a=model_a.fTrain(datas[0][0], datas[0][1], epochs, batch_size, learning_rate,
                             adam=True, val_data=datas[1], verbose=False)
    model_a.fEvaluate(datas[2][0], datas[2][1], key="Test")
    fPlot(history_a)
