import os
import os.path
import torch
import torch.nn as nn
from torch.utils import data
#import pandas as pd
import numpy as np
import torchhd
from torchhd import embeddings

import math
# from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torchhd.functional as functional
import torchhd.embeddings as embeddings



import torchmetrics


class Centroid(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Centroid, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)


    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)

        return functional.cosine_similarity(input, self.weight)


    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)


    @torch.no_grad()
    def add_adapt(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8918974>`_.

        Subtracts the input from the mispredicted class prototype scaled by the learning rate
        and adds the input to the target prototype scaled by the learning rate.
        """
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input, alpha=lr)
        self.weight.index_add_(0, pred, input, alpha=-lr)


    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, alpha1 * input, alpha=lr)
        self.weight.index_add_(0, pred, alpha2 * input, alpha=lr)


    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)

        if torch.isclose(norms, torch.zeros_like(norms), equal_nan=True).any():
            import warnings

            warnings.warn(
                "The norm of a prototype vector is nearly zero upon normalizing, this could indicate a bug."
            )

        norms.clamp_(min=eps)
        self.weight.div_(norms)

    def binarize(self):
        self.weight = Parameter(torch.sign(self.weight), requires_grad=False)


    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )



def AdaBoost(data, model, encode, sample_weights, device):
    tot_error = 0
    whole_indexes = torch.empty(0)
    with torch.no_grad():
        for batch, (samples, labels) in enumerate(data):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv)
            preds = torch.argmax(outputs, dim=-1)
            correct = torch.eq(preds, labels)
            indexes = torch.Tensor( (correct==False).nonzero().squeeze() )
            
            if torch.numel(indexes)>1:
                for i in indexes:
                    tot_error += sample_weights[i + batch*labels.shape[0]]
                indexes = indexes * labels.shape[0]
                whole_indexes = torch.cat((whole_indexes, indexes))

    learner_weight = 0.5 * math.log ( (1 - tot_error)/tot_error )

    for i in range(sample_weights.shape[0]):
        if i in whole_indexes:
            print(i)
            input()
            sample_weights[i] = sample_weights[i] * (math.e ** learner_weight)
        else:
            sample_weights[i] = sample_weights[i] * (math.e ** learner_weight)
    return sample_weights, learner_weight




def Train(data, model, encode, device):
    with torch.no_grad():
        for samples, labels in data:
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            model.add(samples_hv, labels)
    return model

def Retrain(data, model, encode, epoch, lr, device):
    with torch.no_grad():
        for samples, labels in data:
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            model.add_online(samples_hv, labels, lr)
    return model

def Evaluate(data, model, encode, num_classes, device):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    with torch.no_grad():
        for samples, labels in data:
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv)
            accuracy.update(outputs.cpu(), labels)
        print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    return accuracy.compute().item()

def EvaluateBagHD(data, bagHD, encode, num_classes, device):
    avg_acc = 0 #accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    with torch.no_grad():
        for samples, labels in data:
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = []
            for i, learner in enumerate(bagHD):
                outputs.append( learner(samples_hv).cpu().numpy() )
            _scores_final = np.mean(np.array(outputs), axis = 0)
            _preds = np.argmax(_scores_final, axis=-1)
            acc = np.mean(_preds == labels.cpu().numpy())
            avg_acc += acc
        avg_acc /= len(data)
        print(f"Testing accuracy of {(avg_acc * 100):.3f}%")
        # print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    return avg_acc


def EvaluateBoostHD(data, boostHD, encode, learner_weights, device):
    avg_acc = 0 #accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    with torch.no_grad():
        for samples, labels in data:
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = []
            for i, learner in enumerate(boostHD):
                outputs.append( learner(samples_hv).cpu().numpy() * learner_weights[i] )
            _scores_final = np.mean(np.array(outputs), axis = 0)
            _preds = np.argmax(_scores_final, axis=-1)
            acc = np.mean(_preds == labels.cpu().numpy())
            avg_acc += acc
        avg_acc /= len(data)
        print(f"Testing accuracy of {(avg_acc * 100):.3f}%")
        # print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    return avg_acc



class Proj_Encoder(nn.Module):
    def __init__(self, dimensions, num_features, levels, quantize=False):
        super(Proj_Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.quantize = quantize
        # self.device = device
        # self.embed = embeddings.Projection(num_features, dimensions)#.weight.to(device)
        self.embed = embeddings.Sinusoid(num_features, dimensions)
    
    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.embed(x) # for random projection
        if self.quantize:
            sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv


class ID_lev_Encoder(nn.Module):
    def __init__(self, dimensions, num_features, levels, quantize=False):
        super(ID_lev_Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.quantize = quantize
        # self.device = device
        self.base_vectors = embeddings.Random(num_features, dimensions).weight 
        self.levels = embeddings.Level(levels, dimensions)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.base_vectors, self.levels(x))
        sample_hv = torchhd.multiset(sample_hv)
        if self.quantize:
            sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv
