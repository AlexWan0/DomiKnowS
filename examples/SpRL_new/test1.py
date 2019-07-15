import os
import torch
from allennlp.models.model import Model

import os
from regr.graph.allennlp import AllenNlpGraph
from allennlp.data import Vocabulary

from regr.sensor.allennlp.sensor import PhraseSequenceSensor, LabelSensor, LabelSequenceSensor, CartesianProductSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LogisticRegressionLearner
from utils import seed
from config import Config

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from SpRL_reader import SpRLReader as Reader


def data_preparation(config):
    reader = Reader()
    train_dataset = reader.read(os.path.join(config.relative_path, config.train_path))
    valid_dataset = reader.read(os.path.join(config.relative_path, config.train_path))
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    return reader, vocab, train_dataset, valid_dataset


def model_declaration(graph, vocab):
    embedding_dim = 8

    # reset the graph
    graph.detach()

    # retrieve concepts you need in this model
    phrase = graph['linguistic/phrase']

    landmark = graph['application/Landmark']
    trajector = graph['application/Trajector']
    none=graph['application/None']

    # connect sensors and learners
    # features
    phrase['raw'] = PhraseSequenceSensor(vocab, 'sentence', 'phrase')
    phrase['w2v'] = W2VLearner(embedding_dim, phrase['raw'])
    phrase['emb'] = RNNLearner(embedding_dim, phrase['w2v'])

    # phrase label
    landmark['label'] = LabelSequenceSensor('Landmark', output_only=True)
    trajector['label'] = LabelSequenceSensor('Trajector', output_only=True)
    none['label']=LabelSequenceSensor('None',output_only=True)

    # concept prediction
    landmark['label'] = LogisticRegressionLearner(embedding_dim * 2, phrase['emb'])
    trajector['label'] = LogisticRegressionLearner(embedding_dim * 2, phrase['emb'])
    none['label'] = LogisticRegressionLearner(embedding_dim * 2, phrase['emb'])

    # wrap with allennlp model
    lbp = AllenNlpGraph(graph, vocab)
    return lbp


def ontology_declaration():
    from spGraph import splang_Graph
    return splang_Graph


def main():
    # 0.prepare data
    reader, vocab, train_dataset, test_dataset = data_preparation(Config.Data)
    for i in train_dataset:
        print(i)

    graph = ontology_declaration()

    # 1.ontology Declarition
    lbp = model_declaration(graph, vocab)

    # # # 2.5/3. Train and save the model (Explicit inference done automatically)
    # #  # initial the random seeds of all subsystems
    #seed()
    #lbp.train(train_dataset, test_dataset, Config.Train)
    #lbp.save('/tmp/emr')


if __name__ == '__main__':
    main()