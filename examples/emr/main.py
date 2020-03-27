from itertools import chain

import torch
from tqdm import tqdm

from emr.utils import seed
from emr.data import ConllDataLoader
from emr.graph.torch import TorchModel, train, test
from emr.sensor.sensor import DataSensor, LabelSensor, CartesianSensor
from emr.sensor.learner import EmbedderLearner, RNNLearner, MLPLearner, LRLearner

from config import Config as config


def ontology_declaration():
    from graph import graph
    return graph


def model_declaration(graph, vocab, config):
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    pair = graph['linguistic/pair']

    people = graph['application/people']
    organization = graph['application/organization']
    location = graph['application/location']
    other = graph['application/other']
    o = graph['application/O']

    work_for = graph['application/work_for']
    located_in = graph['application/located_in']
    live_in = graph['application/live_in']
    orgbase_on = graph['application/orgbase_on']
    kill = graph['application/kill']

    # feature
    sentence['raw'] = DataSensor('token')
    word['emb'] = EmbedderLearner(sentence['raw'], padding_idx=vocab['token']['_PAD_'], num_embeddings=len(vocab['token']), embedding_dim=50)
    word['ctx_emb'] = RNNLearner(word['emb'], input_size=50, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
    word['feature'] = MLPLearner(word['ctx_emb'], in_features=200, out_features=200)

    pair['emb'] = CartesianSensor(word['ctx_emb'], word['ctx_emb'])
    pair['feature'] = MLPLearner(pair['emb'], in_features=400, out_features=200)

    # label
    word[people] = LabelSensor('Peop')
    word[organization] = LabelSensor('Org')
    word[location] = LabelSensor('Loc')
    word[other] = LabelSensor('Other')
    word[o] = LabelSensor('O')

    word[people] = LRLearner(word['feature'], in_features=200)
    word[organization] = LRLearner(word['feature'], in_features=200)
    word[location] = LRLearner(word['feature'], in_features=200)
    word[other] = LRLearner(word['feature'], in_features=200)
    word[o] = LRLearner(word['feature'], in_features=200)

    # relation
    pair[work_for] = LabelSensor('Work_For')
    pair[live_in] = LabelSensor('Live_In')
    pair[located_in] = LabelSensor('Located_In')
    pair[orgbase_on] = LabelSensor('OrgBased_In')
    pair[kill] = LabelSensor('Kill')

    pair[work_for] = LRLearner(pair['feature'], in_features=200)
    pair[live_in] = LRLearner(pair['feature'], in_features=200)
    pair[located_in] = LRLearner(pair['feature'], in_features=200)
    pair[orgbase_on] = LRLearner(pair['feature'], in_features=200)
    pair[kill] = LRLearner(pair['feature'], in_features=200)

    # program
    lbp = TorchModel(graph, **config)
    return lbp


def print_result(model, epoch=None, phase=None):
    header = ''
    if epoch is not None:
        header += 'Epoch {} '.format(epoch)
    if phase is not None:
        header += '{} '.format(phase)
    print('{}Loss:'.format(header))
    loss = model.loss.value()
    for (pred, _), value in loss.items():
        print(' - ', pred.sup.prop_name.name, value.item())
    print('{}Metrics:'.format(header))
    metrics = model.metric.value()
    for (pred, _), value in metrics.items():
        print(' - ', pred.sup.prop_name.name, str({k: v.item() for k, v in value.items()}))


def main():
    graph = ontology_declaration()

    training_set = ConllDataLoader(config.Data.train_path,
                                   batch_size=config.Train.batch_size,
                                   skip_none=config.Data.skip_none)
    valid_set = ConllDataLoader(config.Data.valid_path,
                                batch_size=config.Train.batch_size,
                                skip_none=config.Data.skip_none,
                                vocab=training_set.vocab)

    lbp = model_declaration(graph, training_set.vocab, config.Model)
    lbp.cuda()

    seed()
    opt = torch.optim.Adam(lbp.parameters())
    for epoch in range(10):
        print('Epoch:', epoch)
        print('Training:')
        for _ in tqdm(train(lbp, training_set, opt), total=len(training_set)):
            pass
        print_result(lbp, epoch, 'Training')

        print('Validation:')
        for _ in tqdm(test(lbp, valid_set), total=len(valid_set)):
            pass
        print_result(lbp, epoch, 'Validation')

if __name__ == '__main__':
    main()
