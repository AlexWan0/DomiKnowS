from typing import List, Dict, Any, NoReturn
from collections import OrderedDict
import torch
from torch.nn import Module
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from ...utils import prod
from ...graph import Property
from .base import ReaderSensor, ModuleSensor, SinglePreMaskedSensor, MaskedSensor, PreArgsModuleSensor


class SentenceSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=False
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_dim=(), output_only=output_only) # *pres=[]
        self.embedders = OrderedDict() # list of SentenceEmbedderLearner

    def add_embedder(self, key, embedder):
        self.reader.claim(key, embedder)
        self.embedders[key] = embedder

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        # This sensor it self can do nothing
        # mayby with self.embedders something more can happen?
        return None


class LabelSensor(ReaderSensor):
    def __init__(
        self,
        reader,
        key: str,
        output_only: bool=True
    ) -> NoReturn:
        ReaderSensor.__init__(self, reader, key, output_dim=(), output_only=output_only)


class ConcatSensor(PreArgsModuleSensor, MaskedSensor):
    class Concat(Module):
        def forward(self, *x):
            # TODO: flatten
            return torch.cat(x, dim=-1)

    def create_module(self):
        return ConcatSensor.Concat()

    def update_output_dim(self):
        output_dim = 0
        for pre_dim in self.pre_dims:
            if len(pre_dim) == 0:
                output_dim += 1
            else:
                output_dim += prod(pre_dim) # assume flatten
        self.output_dim = (output_dim,)

    def get_mask(self, context: Dict[str, Any]):
        for pre in self.pres:
            for name, sensor in pre.find(MaskedSensor):
                return sensor.get_mask(context)
            else:
                # not found
                continue
            # found
            break
        else:
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        return None # not going to here


class CartesianProductSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    class CP(Module):
        def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
            # TODO: flatten
            xs = x.size()
            ys = y.size()
            assert xs[0] == ys[0]
            # torch cat is not broadcasting, do repeat manually
            xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
            yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
            return torch.cat([xx, yy], dim=3)

    class SelfCP(CP):
        def forward(self, x):
            return CartesianProductSensor.CP.forward(self, x, x)

    def create_module(self):
        return CartesianProductSensor.SelfCP()

    def update_output_dim(self):
        if len(self.pre_dim) == 0:
            output_dim = 2
        else:
            output_dim = prod(self.pre_dim) * 2 # assume flatten
        self.output_dim = (output_dim,)

    def __init__(
        self,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.pre = pre
        PreArgsModuleSensor.__init__(self, pre, output_only=output_only)


    def get_mask(self, context: Dict[str, Any]):
        for name, sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(context).float()
        ms = mask.size()
        mask = mask.view(ms[0], ms[1], 1).matmul(
            mask.view(ms[0], 1, ms[1]))  # (b,l,l)
        return mask


class SentenceEmbedderSensor(SinglePreMaskedSensor, ModuleSensor):
    def create_module(self):
        self.embedding = Embedding(
            num_embeddings=0, # later load or extend
            embedding_dim=self.embedding_dim,
            pretrained_file=self.pretrained_file,
            vocab_namespace=self.key,
            trainable=False,
        )
        return BasicTextFieldEmbedder({self.key: self.embedding})

    def __init__(
        self,
        key: str,
        embedding_dim: int,
        pre,
        pretrained_file: str=None,
        output_only: bool=False
    ) -> NoReturn:
        self.key = key
        self.embedding_dim = embedding_dim
        self.pretrained_file = pretrained_file
        self.pre = pre
        ModuleSensor.__init__(self, pre, output_only=output_only)

        for name, pre_sensor in pre.find(SentenceSensor):
            pre_sensor.add_embedder(key, self)
            self.tokens_key = pre_sensor.key # used by reader.update_textfield()
            break
        else:
            raise TypeError()

    def update_context(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if self.fullname in context and isinstance(context[self.fullname], dict):
            context[self.fullname + '_index'] = context[self.fullname] # reserve
            force = True
        return SinglePreMaskedSensor.update_context(self, context, force)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return self.module(context[self.fullname])

    def get_mask(self, context: Dict[str, Any]):
        # TODO: make sure update_context has been called
        return get_text_field_mask(context[self.fullname + '_index'])


class NGramSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    class NGram(Module):
        def __init__(self, ngram):
            Module.__init__(self)
            self.ngram = ngram

        def forward(self, x):
            #import pdb; pdb.set_trace()
            shifted = []
            size = x.size() # (b, l, c)
            for i in torch.arange(self.ngram):
                shifted_x = torch.zeros((size[0], size[1]+self.ngram, size[2]), device=x.device)
                shifted_x[:, i:i-self.ngram, :] = x
                shifted.append(shifted_x)
            new_x = torch.cat(shifted, dim=-1)
            offset = int((self.ngram-1) / 2)
            return new_x[:, offset:offset-self.ngram, :]

    def create_module(self):
        return NGramSensor.NGram(self.ngram)

    def update_output_dim(self):
        #import pdb; pdb.set_trace()
        self.output_dim = tuple([dim * self.ngram for dim in self.pre_dim])

    def __init__(
        self,
        ngram: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.ngram = ngram
        self.pre = pre
        PreArgsModuleSensor.__init__(self, pre, output_only=output_only)


class TokenDistantSensor(PreArgsModuleSensor, SinglePreMaskedSensor):
    def find_base(s, n):
        length = lambda b: (1 - b ** (n + 1)) / (1 - b)
        res = minimize_scalar(lambda b : (length(b) - s) ** 2, method='bounded', bounds=(1, (s-1)**(1./n)))
        return res.x

    class Dist(Module):
        def __init__(self, emb_num, window):
            Module.__init__(self)
            self.emb_num = emb_num # must define emb_num (to have lb and ub) before window
            self.window = window

        @property
        def window(self):
            return self._window

        @window.setter
        def window(self, window):
            self._window = window
            ul = np.floor(window / 2)
            self._base = TokenDistantSensor.find_base(ul, self.ub - 1)

        @property
        def base(self):
            return self._base

        @property
        def emb_num(self):
            return self._emb_num

        @emb_num.setter
        def emb_num(self, emb_num):
            self._emb_num = emb_num
            self._lb = -np.floor((emb_num - 1) / 2)
            self._ub = np.ceil((emb_num - 1) / 2)

        @property
        def lb(self):
            return self._lb

        @property
        def ub(self):
            return self._ub

        def forward(self, x):
            batch = x.shape[0]
            length = x.shape[1]
            #(l*2)
            dist = torch.arange(-length + 1, length, device=x.device)
            rows = []
            for i in range(length):
                rows.append(dist[i:i + length].view(1, -1))
            #(l, l)
            dist = torch.cat(tuple(reversed(rows)))
            #print(dist)
            sign = dist.sign()
            dist = dist.abs()
            dist = dist.to(dtype=torch.float)
            dist = (dist.log() / np.log(self.base) + 1).floor()
            #print(dist)
            dist[dist < 0] = 0
            dist = dist * sign.to(dtype=dist.dtype, device=dist.device)
            dist[dist < self.lb] = self.lb
            dist[dist > self.ub] = self.ub
            dist = dist - self.lb
            dist = dist.to(dtype=torch.long)
            #print(dist)
            #(n, n)
            eye = torch.eye(self.emb_num, device=dist.device)
            #(l*l, n)
            dist = dist.view(-1, 1).repeat(1, self.emb_num)
            #(l*l, n)
            dist = eye.gather(0, dist)
            #(l, l, n)
            dist = dist.view(length, length, self.emb_num)
            #print(dist)
            #(b, l, l, n)
            dist = dist.view(1, length, length, self.emb_num).repeat(batch, 1, 1, 1)
            return dist

    def create_module(self):
        return TokenDistantSensor.Dist(self.emb_num, self.window)

    def update_output_dim(self):
        self.output_dim = (self.emb_num,)

    def __init__(
        self,
        emb_num: int,
        window: int,
        pre: Property,
        output_only: bool=False
    ) -> NoReturn:
        self.pre = pre
        self.emb_num = emb_num
        self.window = window
        PreArgsModuleSensor.__init__(self, pre, output_only=output_only)


    def get_mask(self, context: Dict[str, Any]):
        for name, sensor in self.pre.find(MaskedSensor):
            break
        else:
            print(self.pre)
            raise RuntimeError('{} require at least one pre-required sensor to be MaskedSensor.'.format(self.fullname))

        mask = sensor.get_mask(context).float()
        ms = mask.size()
        mask = mask.view(ms[0], ms[1], 1).matmul(
            mask.view(ms[0], 1, ms[1]))  # (b,l,l)
        return mask
