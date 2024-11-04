import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import *
import torchvision.transforms as T
from alive_progress import alive_bar
import argparse as ap
import warnings
from boosthd.torchhd_models import *

import torch

warnings.filterwarnings("ignore")




def boosting(bsize, dim, enc, bw, epochs, lr, nlearners, dataset, binary, key):

    train_ds, train_dl, test_dl, _tr_len, _ts_len, fdim, num_classes = load_dataset(dataset, bsize)

    if enc == "rff":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary)
    elif enc == "proj":
        encoder = Proj_Encoder(dim, fdim, binary)
    elif enc == "idlev":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary)

    encoder = encoder.to(device)
    boostHD = [Centroid(dim, num_classes).to(device) for _ in range(nlearners)]
    sample_weights = torch.full((len(train_ds),), 1/(len(train_ds)))
    learner_weights = []

    for i in range(nlearners):
        train_dl = boosting_train_load_dataset(train_ds, sample_weights, bsize)
        Train(train_dl, boostHD[i], encoder, device)
        for epoch in range(epochs):
            Retrain(train_dl, boostHD[i], encoder, epoch, 20, device)

        if(binary):
                boostHD[i].binarize()
        
        sample_weights, learner_weight = AdaBoost(test_dl, boostHD[i], encoder, sample_weights, device)
        learner_weights.append(learner_weight)

    avg_acc = EvaluateBoostHD(test_dl, boostHD, encoder, learner_weights, device)

    f = open("boosting.txt", "a")
    f.write("Dataset: {} - Learners: {} - Enc: {} - Dim: {} - Bin: {} - Accuracy: {}\n".format(dataset, nlearners, enc, dim, binary, avg_acc))
    f.close()







def bagging(bsize, dim, bw, epochs, enc, dataset, binary, nlearners, device):
    
    train_ds, train_dl, test_dl, _tr_len, _ts_len, fdim, num_classes = load_dataset(dataset, bsize)

    if enc == "rff":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary)
    elif enc == "proj":
        encoder = Proj_Encoder(dim, fdim, binary)
    elif enc == "idlev":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary)

    encoder = encoder.to(device)

    # encoder = [Proj_Encoder(dim, fdim, binary).to(device) for _ in range(nlearners)]
    bagHD = [Centroid(dim, num_classes).to(device) for _ in range(nlearners)]

    for i in range(nlearners):
        train_dl = bagging_train_load_dataset(train_ds, bsize)
        Train(train_dl, bagHD[i], encoder, device)
        for epoch in range(epochs):
            Retrain(train_dl, bagHD[i], encoder, epoch, 20, device)
            Evaluate(test_dl, bagHD[i], encoder, num_classes, device)

        # bagHD[i].normalize()

        if(binary):
            bagHD[i].binarize()

    avg_acc = EvaluateBagHD(test_dl, bagHD, encoder, num_classes, device)

    f = open("bagging.txt", "a")
    f.write("Dataset: {} - Learners: {} - Enc: {} - Dim: {} - Bin: {} - Accuracy: {}\n".format(dataset, nlearners, enc, dim, binary, avg_acc))
    f.close()




def baselineHD(bsize, dim, bw, epochs, enc, dataset, binary, device):
    
    train_ds, train_dl, test_dl, _tr_len, _ts_len, fdim, num_classes = load_dataset(dataset, bsize)

    if enc == "rff":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary, device)
    elif enc == "proj":
        encoder = Proj_Encoder(dim, fdim, binary, device)
    elif enc == "idlev":
        encoder = ID_lev_Encoder(dim, fdim, 256, binary, device)

    encoder = encoder.to(device)

    model = Centroid(dim, num_classes)
    model = model.to(device)

    Train(train_dl, model, encoder, device)
    for epoch in range(epochs):
        Retrain(train_dl, model, encoder, epoch, 0.5, device)

    if(binary):
        model.binarize()
    
    avg_acc = Evaluate(test_dl, model, encoder, num_classes, device)

    f = open("baseline.txt", "a")
    f.write("Dataset: {} - Enc: {} - Dim: {} - Bin: {} - Accuracy: {}\n".format(dataset, enc, dim, binary, avg_acc))
    f.close()


if __name__ == '__main__':
    ##not needed if running full experiments set as below
    parser = ap.ArgumentParser()
    parser.add_argument('--bsize',      type=int,   default=2, required=False)
    parser.add_argument('--dim',        type=int,   default=10000, required=False)
    parser.add_argument('--bw',         type=float, default=1.0, required=False)
    parser.add_argument('--epochs',     type=int,   default=1, required=False)
    parser.add_argument('--enc',        type=str,   default="proj", required=False)
    parser.add_argument('--dataset',    type=str,   default="mnist", required=False)
    parser.add_argument('--lr',         type=float, default=1e-3, required=False)
    parser.add_argument('--nlearners',  type=int,   default=2, required=False)
    parser.add_argument('--method',     type=str,   default="base", required=False)
    parser.add_argument('--binary',     action='store_true')
    args = parser.parse_args()



    datasets = ["cardio"] #,"cardio", "ucihar", "isolet"]  #"fmnist", , "ucihar", "isolet", "mnist", "cardio", "fmnist",
    dimensions = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] #
    encodings = ["proj"] #rff, idlev
    binary = [False, True]
    learners = [2, 5, 10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    for dataset in datasets:
        for enc in encodings:
            for l in learners:
                for b in binary:
                    for dim in dimensions:
                        bagging(128, dim, 520, 20, enc, dataset, b, l, device)

    for dataset in datasets:
        for enc in encodings:
            for l in learners:
                for b in binary:
                    for dim in dimensions:
                        boosting(128, dim, enc, 520, 20, 1e-3, l, dataset, b, device)

    for dataset in datasets:
        for enc in encodings:
            for b in binary:
                for dim in dimensions:
                    baselineHD(1024, dim, 520, 20, enc, dataset, b, device)

    

    