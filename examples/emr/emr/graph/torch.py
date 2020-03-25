import abc
from collections import defaultdict
from typing import Any
from itertools import combinations

import torch
from torch.nn import functional as F

from regr.graph.property import Property
from emr.sensor.learner import TorchSensor, ModuleLearner


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class CMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, weight=None):
        if weight is None:
            weight = torch.ones_like(input, dtype=torch.bool)
        preds = (input > 0).clone().detach().bool()
        labels = target.clone().detach().bool()
        tp = (preds * labels * weight).sum()
        fp = (preds * (~labels) * weight).sum()
        tn = ((~preds) * (~labels) * weight).sum()
        fn = ((~preds) * labels * weight).sum()
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


class PRF1WithLogitsMetric(CMWithLogitsMetric):
    def forward(self, input, target, weight=None):
        CM = super().forward(input, target, weight)
        tp = CM['TP'].float()
        fp = CM['FP'].float()
        fn = CM['FN'].float()
        if CM['TP']:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
        return {'P': p, 'R': r, 'F1': f1}


class MetricTracker(torch.nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.list = []
        self.dict = defaultdict(list)

    def reset(self):
        self.list.clear()
        self.dict.clear()

    def __call__(self, *args, **kwargs) -> Any:
        value = self.metric(*args, **kwargs)
        self.list.append(value)
        return value

    def __call_dict__(self, keys, *args, **kwargs) -> Any:
        value = self.metric(*args, **kwargs)
        self.dict[keys].append(value)
        return value

    def __getitem__(self, keys):
        return lambda *args, **kwargs: self.__call_dict__(keys, *args, **kwargs)

    def value(self, reset=False):
        if self.list and self.dict:
            raise RuntimeError('%s cannot be used as list-like and dict-like the same time.', str(type(self)))
        if self.list:
            value = wrap_batch(self.list)
            value = super().__call__(value)
        elif self.dict:
            #value = wrap_batch(self.dict)
            #value = super().__call__(value)
            func = super().__call__
            value = {k: func(v) for k, v in self.dict.items()}
        else:
            value = None
        if reset:
            self.reset()
        return value


class MacroAverageTracker(MetricTracker):
    def forward(self, values):
        def func(value):
            return value.clone().detach().mean()
        def apply(value):
            if isinstance(value, dict):
                return {k: apply(v) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                return func(value)
            else:
                return apply(torch.tensor(value))
        retval = apply(values)
        return retval

class PRF1Tracker(MetricTracker):
    def __init__(self):
        super().__init__(CMWithLogitsMetric())

    def forward(self, values):
        CM = wrap_batch(values)
        tp = CM['TP'].sum().float()
        fp = CM['FP'].sum().float()
        fn = CM['FN'].sum().float()
        if tp:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
        return {'P': p, 'R': r, 'F1': f1}

class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.loss = MacroAverageTracker(BCEWithLogitsLoss())
        #self.metric = MacroAverageTracker(PRF1WithLogitsMetric())
        self.metric = PRF1Tracker()

        def func(node):
            if isinstance(node, Property):
                return node
            return None
        for node in self.graph.traversal_apply(func):
            for _, sensor in node.find(ModuleLearner):
                self.add_module(sensor.fullname, sensor.module)

    def move(self, value, device=None):
        device = device or next(self.parameters()).device
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, list):
            return [self.move(v, device) for v in value]
        elif isinstance(value, tuple):
            return (self.move(v, device) for v in value)
        elif isinstance(value, dict):
            return {k: self.move(v, device) for k, v in value.items()}
        else:
            raise NotImplementedError('%s is not supported. Can only move list, dict of tensors.', type(value))

    def forward(self, data):
        data = self.move(data)
        loss = 0
        metric = {}
        def all_properties(node):
            if isinstance(node, Property):
                return node
        for prop in self.graph.traversal_apply(all_properties):
            for (_, sensor1), (_, sensor2) in combinations(prop.find(TorchSensor), r=2):
                if sensor1.target:
                    target_sensor = sensor1
                    output_sensor = sensor2
                elif sensor2.target:
                    target_sensor = sensor2
                    output_sensor = sensor1
                else:
                    # TODO: should different learners get closer?
                    continue
                if output_sensor.target:
                    # two targets, skip
                    continue
                logit = output_sensor(data)
                logit = logit.squeeze()
                mask = output_sensor.mask(data)
                labels = target_sensor(data)
                labels = labels.float()
                if self.loss:
                    local_loss = self.loss(logit, labels, mask)
                    loss += local_loss
                if self.metric:
                    local_metric = self.metric[output_sensor, target_sensor](logit, labels, mask)
                    metric[output_sensor, target_sensor] = local_metric
        return loss, metric, data


def dict_zip(*dicts, fillvalue=None):  # https://codereview.stackexchange.com/a/160584
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d.get(k, fillvalue) for d in dicts] for k in all_keys}


def wrap_batch(values, fillvalue=0):
    if isinstance(values, (list, tuple)):
        if isinstance(values[0], dict):
            values = dict_zip(*values, fillvalue=fillvalue)
            values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
        elif isinstance(values[0], torch.Tensor):
            values = torch.stack(values)
    elif isinstance(values, dict):
        values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
    return values


def train(model, dataset, opt):
    model.train()
    model.loss.reset()
    model.metric.reset()
    for data in dataset:
        opt.zero_grad()
        loss, metric, output = model(data)
        loss.backward()
        opt.step()
        yield loss, metric, output


def test(model, dataset):
    model.eval()
    model.loss.reset()
    model.metric.reset()
    with torch.no_grad():
        for data in dataset:
            loss, metric, output = model(data)
            yield loss, metric, output


def eval_many(model, dataset):
    model.eval()
    model.loss.reset()
    model.metric.reset()
    with torch.no_grad():
        for data in dataset:
            _, _, output = model(data)
            yield output


def eval_one(model, data):
    # TODO: extend one sample data to 1-batch data
    model.eval()
    model.loss.reset()
    model.metric.reset()
    with torch.no_grad():
        _, _, output = model(data)
        return output
