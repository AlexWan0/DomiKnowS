from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from ..base import AutoNamed
from ..utils import wrap_batch


class BinaryCMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, data_item, prop, weight=None, dim=None):
        if weight is None:
            weight = torch.tensor(1, device=input.device)
        preds = F.softmax(input, dim=-1)
        preds = (preds > 0.5).clone().detach().to(dtype=weight.dtype)
        labels = target.clone().detach().to(dtype=weight.dtype, device=input.device)
        assert (0 <= labels).all() and (labels <= 1).all()
        tp = (preds * labels * weight)[:, 1].sum()
        fp = (preds * (1 - labels) * weight)[:, 1].sum()
        tn = (preds * labels * weight)[:, 0].sum()
        fn = ((1 - preds) * labels * weight)[:, 1].sum()
#         print(prop.name)
#         datanode = data_item.getDataNode()
#         result = datanode.getInferMetric(inferType='argmax')
#         val =  result[str(prop.name)]
#         print(val)
#         return {"TP": val["TP"], 'FP': val["FP"], 'TN': val["TN"], 'FN': val["FN"]}
#         print(preds, labels, {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

class DatanodeCMMetric(torch.nn.Module):
    def forward(self, input, target, data_item, prop, weight=None, dim=None):
        datanode = data_item.getDataNode()
        result = datanode.getInferMetric(inferType='ILP')
        val =  result[str(prop.name)]
        return {"TP": val["TP"], 'FP': val["FP"], 'TN': val["TN"], 'FN': val["FN"]}

class CMWithLogitsMetric(BinaryCMWithLogitsMetric):
    def forward(self, input, target, data_item, prop, weight=None):
        num_classes = input.shape[-1]
        input = input.view(-1, num_classes)
        target = target.view(-1).to(dtype=torch.long)
        target = F.one_hot(target.view(-1), num_classes=num_classes)
        return super().forward(input, target, data_item, prop, weight)


class BinaryPRF1WithLogitsMetric(BinaryCMWithLogitsMetric):
    def forward(self, input, target, data_item, prop, weight=None):
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

class PRF1WithLogitsMetric(CMWithLogitsMetric, BinaryPRF1WithLogitsMetric):
    pass


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

    def kprint(self, k):
        if (
            isinstance(k, tuple) and
            len(k) == 2 and
            isinstance(k[0], AutoNamed) and 
            isinstance(k[1], AutoNamed)):
            return k[0].sup.name.name
        else:
            return k

    def value(self, reset=False):
        if self.list and self.dict:
            raise RuntimeError('{} cannot be used as list-like and dict-like the same time.'.format(type(self)))
        if self.list:
            value = wrap_batch(self.list)
            value = super().__call__(value)
        elif self.dict:
            #value = wrap_batch(self.dict)
            #value = super().__call__(value)
            func = super().__call__
            value = {self.kprint(k): func(v) for k, v in self.dict.items()}
        else:
            value = None
        if reset:
            self.reset()
        return value

    def __str__(self):
        return str(self.value())


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


class ValueTracker(MetricTracker):
    def forward(self, values):
        return values


class PRF1Tracker(MetricTracker):
    def __init__(self, metric=CMWithLogitsMetric()):
        super().__init__(metric)

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


class BinaryPRF1Tracker(PRF1Tracker):
    def __init__(self, metric=BinaryCMWithLogitsMetric()):
        super().__init__(metric)
