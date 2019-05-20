from typing import Dict, Iterable
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary


#from typing import Any, Union
from typing import List, Tuple, Dict, Callable
from allennlp.models.model import Model
import torch
from torch import Tensor
from torch.nn import Module
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from regr.graph import Graph, Concept

from regr.scaffold.allennlp import ModuleFunc, DataInstance


def datainput(
    data_name: str
) -> Tuple[Module, ModuleFunc]:
    def func(data: DataInstance) -> Tensor:
        tensor = data[data_name]
        return tensor

    return None, func


def word2vec(
    input_func: ModuleFunc,
    num_embeddings: int,
    embedding_dim: int,
    token_name: str,
) -> Tuple[Module, ModuleFunc]:
    (module, input_func), conf = input_func[0]

    # token_name='tokens' is from data reader, name of TokenIndexer
    # seq_name='sentence' is from data reader, name of TextField
    # quite confusing, TODO: real want to get rid of them
    token_embedding = Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({
        token_name: token_embedding})
    dropout = torch.nn.Dropout(0.5)
    dropout.add_module('emb', word_embeddings)
    if module is not None:
        # add submodule
        # TODO: move to wrapper or concept assignment?
        word_embeddings.add_module('sub', module)

    def func(data: DataInstance) -> Tensor:
        tensor = input_func(data)  # input_func is tuple(func, conf)
        tensor = dropout(word_embeddings(tensor))
        return tensor

    return dropout, func


def word2vec_rnn(
    input_func: ModuleFunc,
    num_embeddings: int,
    embedding_dim: int,
    token_name: str,
) -> Tuple[Module, ModuleFunc]:
    (module, input_func), conf = input_func[0]

    # token_name='tokens' is from data reader, name of TokenIndexer
    # seq_name='sentence' is from data reader, name of TextField
    # quite confusing, TODO: real want to get rid of them
    token_embedding = Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({
        token_name: token_embedding})
    from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
    rnn = PytorchSeq2SeqWrapper(torch.nn.GRU(embedding_dim, embedding_dim, batch_first=True, dropout=0.5, bidirectional=True))
    dropout = torch.nn.Dropout(0.5)
    rnn.add_module('emb', word_embeddings)
    dropout.add_module('rnn', rnn)
    if module is not None:
        # add submodule
        # TODO: move to wrapper or concept assignment?
        word_embeddings.add_module('sub', module)

    def func(data: DataInstance) -> Tensor:
        tensor = input_func(data)  # input_func is tuple(func, conf)
        tensor = dropout(rnn(word_embeddings(tensor), data['mask']))
        return tensor

    return dropout, func

class Cpcat(Module):
    def __init__(self):
        Module.__init__(self)

    def forward(self, x, y):  # (b,l1,f1) x (b,l2,f2) -> (b, l1, l2, f1+f2)
        xs = x.size()
        ys = y.size()
        assert xs[0] == ys[0]
        # torch cat is not broadcasting, do repeat manually
        xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
        yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
        return torch.cat([xx, yy], dim=3)


def cartesianprod_concat(
    input_func: ModuleFunc
) -> Tuple[Module, ModuleFunc]:
    (module, input_func), conf = input_func[0]

    cpcat = Cpcat()
    if module is not None:
        # add submodule
        # TODO: move to wrapper or concept assignment?
        cpcat.add_module('sub', module)

    def func(data: DataInstance) -> Tensor:
        tensor = input_func(data)  # input_func is tuple(func, conf)
        tensor = cpcat(tensor, tensor)
        return tensor

    return cpcat, func


def fullyconnected(
    input_func: ModuleFunc,
    input_dim: int,
    label_dim: int,
) -> Tuple[Module, ModuleFunc]:
    (module, input_func), conf = input_func[0]

    fc = torch.nn.Linear(
        in_features=input_dim,
        out_features=label_dim)
    if module is not None:
        # add submodule
        # TODO: move to wrapper or concept assignment?
        fc.add_module('sub', module)

    def func(data: DataInstance) -> Tensor:
        tensor = input_func(data)
        tensor = fc(tensor)
        return tensor

    return fc, func

def logsm(
    input_func: ModuleFunc,
    input_dim: int,
    label_dim: int,
) -> Tuple[Module, ModuleFunc]:
    (module, input_func), conf = input_func[0]

    fc = torch.nn.Linear(
        in_features=input_dim,
        out_features=label_dim)
    sm = torch.nn.LogSoftmax(dim=-1)

    sm.add_module('sub', fc)
    if module is not None:
        # add submodule
        # TODO: move to wrapper or concept assignment?
        fc.add_module('sub', module)

    def func(data: DataInstance) -> Tensor:
        tensor = input_func(data)
        tensor = sm(fc(tensor))
        return tensor

    return sm, func

from regr.graph import Graph
from regr.scaffold import Scaffold
from regr.scaffold.allennlp import BaseModel
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from torch.optim import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from data import Data
else:
    # uses current package visibility
    from .data import Data


DEBUG_TRAINING = False


def get_trainer(
    graph: Graph,
    model: BaseModel,
    data: Data,
    scaffold: Scaffold,
    lr=1., wd=0.003, batch=64, epoch=1000, patience=50
) -> Trainer:
    # get the loss
    model.loss_func = scaffold.get_loss(graph, model)

    # prepare GPU
    if torch.cuda.is_available() and not DEBUG_TRAINING:
        device = 0
        model = model.cuda()
    else:
        device = -1

    # prepare optimizer
    #print([p.size() for p in model.parameters()])
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd) # SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
    iterator = BucketIterator(batch_size=batch,
                              sorting_keys=[('sentence', 'num_tokens')],
                              track_epoch=True)
    iterator.index_with(model.vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=data.train_dataset,
                      validation_dataset=data.valid_dataset,
                      patience=patience,
                      num_epochs=epoch,
                      cuda_device=device)

    return trainer
