'''
# Example: Entity-Mention-Relation (EMR)
## Steps
This example follows the three declarative steps:
1. Knowledge Declaration
2. Data Declaration
3. Learning Declaration
'''

#### With `regr`, we assign sensors to properties of concept.
#### There are two types of sensor: `Sensor`s and `Learner`s.
#### `Sensor` is the more general term, while a `Learner` is a `Sensor` with learnable parameters.
from regr.sensor.allennlp.sensor import SentenceSensor, SentenceEmbedderSensor, LabelSensor, CartesianProductSensor, ConcatSensor, NGramSensor, TokenDistantSensor, TokenDepSensor, TokenLcaSensor, TokenDepDistSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, MLPLearner, ConvLearner, LogisticRegressionLearner

#### `AllenNlpGraph` is a special subclass of `Graph` that wraps a `Graph` and adds computational functionalities to it.
from regr.graph.allennlp import AllenNlpGraph

#### There are a few other components that are needed in common machine learning models.
#### * `Conll04SensorReader` is a AllenNLP `DatasetReader`.
#### We added a bit of magic to make it more compatible with `regr`.
#### See `data.py` for details.
#### * `Config` contains configurations for model, data, and training.
#### * `seed` is a useful function that resets random seed of all involving sub-systems: Python, numpy, and PyTorch, to make the performance of training consistent, as a demo.
#from .data import Conll04SensorReader as Reader
from data_spacy import Conll04SpaCyBinaryReader as Reader
from config import Config
from utils import seed


#### "*Knowledge Declaration*"
#### A graph of the concepts, representing the ontology in this application, is declared.
#### It can be compile from standard ontology formats like `OWL`, writen with python grammar directly, or combine both way.
#### Here we just import the graph from `graph.py`.
#### Please also refer to `graph.py` for details.
def knowledge_declaration():
    from graph import graph
    return graph


#### "*Data Declaration*" and "*Learning Declaration*"
#### Sensors and learners are connected to the graph, what wraps the graph with functionalities to retieve data, forward computing, learning from error, and inference during all those processes.
#### `graph` is a `Graph` object retrieved from the "Knowledge Declaration".
#### `config` is configurations relatred to the model.
def data_and_learning_declaration(graph, config):
    #### `Graph` class has some kind of global variables.
    #### Use `.detach()` to reset them to avoid baffling error.
    graph.detach()

    #### Retrieve concepts that are needed in this model.
    #### Notice that these concepts are already well defined in `graph.py`.
    #### Here we just retrieve them to use them as python variables.
    #### `sentence`, `phrase`, and `pair` are basic linguistic concepts in this demo.
    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    pair = graph['linguistic/pair']

    #### `people` and `organization` are entities we want to extract in this demo.
    people = graph['application/people']
    organization = graph['application/organization']

    #### `kill` and `work_for` are relations we want to extract in this demo.
    work_for = graph['application/work_for']
    kill = graph['application/kill']

    #### Create a `Reader` instance, to be assigned with properties, and allow the model to get corresponding data from it.
    reader = Reader()

    #### "*Data Declaration*"
    #### We start with linguistic concepts.
    #### `SentenceSensor` provides the ability to read words from a `TextField` in AllenNLP.
    #### It takes two arguments, firstly the reader to read with, and secondly a `key` for the reader to refer to correct `TextField`.
    sentence['raw'] = SentenceSensor(reader, 'sentence')
    #### `SentenceEmbedderSensor` provides the ability to index the words and convert them into word embeddings.
    #### Notice that `SentenceEmbedderSensor` is a load pretrained embedding and do not train anymore.
    word['raw'] = SentenceEmbedderSensor('word', config.pretrained_dims['word'], sentence['raw'], pretrained_file=config.pretrained_files['word'])
    #### We can also specify `SentenceEmbedderLearner` to train tokens with or without pretrained parameters.
    #### Here `pos_tag` and `dep_tag` are also text-based in the original input.
    word['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])
    word['dep'] = SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    # possible to add more this kind
    #### Then we can concatenate them together.
    word['all'] = ConcatSensor(word['raw'], word['pos'], word['dep'])
    #### `NGramSensor` use a sliding window to collect context for each word.
    word['ngram'] = NGramSensor(config.ngram, word['all'])
    #### `RNNLearner` takes a sequence of representations as input, encodes them with recurrent nerual networks (RNN), like LSTM or GRU, and provides the encoded output.
    word['encode'] = RNNLearner(word['ngram'], layers=config.rnn.layers, bidirectional=config.rnn.bidirectional, dropout=config.dropout)
    #### The output is high-dimensional after N-gram and bidirectional RNN. We want to have a compact representation with a simple fully connected layer.
    word['compact'] = MLPLearner(config.compact.layers, word['encode'], activation=config.activation)
    #### `CartesianProductSensor` is a `Sensor` that takes the representation from `word['emb']`, makes all possible combination of them, and generates a concatenating result for each combination.
    pair['cat'] = CartesianProductSensor(word['compact'])
    #### Also add some pair-wise features.
    pair['tkn_dist'] = TokenDistantSensor(config.distance_emb_size * 2, config.max_distance, sentence['raw'])
    pair['tkn_dep'] = TokenDepSensor(sentence['raw'])
    pair['tkn_dep_dist'] = TokenDepDistSensor(config.distance_emb_size, config.max_distance, sentence['raw'])
    #### Map the onehot features to a dense feature space by a simple fully connected layer.
    pair['onehots'] = ConcatSensor(pair['tkn_dist'], pair['tkn_dep'], pair['tkn_dep_dist'])
    pair['emb'] = MLPLearner([config.relemb.emb_size,], pair['onehots'], activation=None)
    #### Yet another pair-wise feature.
    pair['tkn_lca'] = TokenLcaSensor(sentence['raw'], word['compact'])
    #### Put them all together.
    pair['encode'] = ConcatSensor(pair['cat'], pair['tkn_lca'], pair['emb'])

    #### Then we connect properties with ground-truth from `reader`.
    #### `LabelSensor` takes the `reader` as argument to provide the ground-truth data.
    #### The second argument indicates the key we used for each lable in reader.
    #### The last keyword argument `output_only` indicates that these sensors are not to be used with forward computation.
    people['label'] = LabelSensor(reader, 'Peop', output_only=True)
    organization['label'] = LabelSensor(reader, 'Org', output_only=True)

    #### We connect properties with learners that generate predictions.
    #### Notice that we connect the predicting `Learner`s to the same properties as "ground-truth" `Sensor`s.
    #### Multiple assignment is a feature in `regr` to allow each property to have multiple sources.
    #### Value from different sources will be compared, to generate inconsistency error.
    #### The training of this model is then based on this inconsistency error.
    #### In this example, "ground-truth" `Sensor`s has no parameters to be trained, while predicting `Learner`s have all sets of paramters to be trained.
    #### The error also propagate backward through the computational path to all modules as assigned above.
    people['label'] = LogisticRegressionLearner(word['encode'])
    organization['label'] = LogisticRegressionLearner(word['encode'])

    #### We repeat these on composed-concepts.
    #### There is nothing different in usage thought they are higher ordered concepts.
    work_for['label'] = LabelSensor(reader, 'Work_For', output_only=True)
    kill['label'] = LabelSensor(reader, 'Kill', output_only=True)

    #### We also connect the predictors for composed-concepts.
    work_for['label'] = LogisticRegressionLearner(pair['encode'])
    kill['label'] = LogisticRegressionLearner(pair['encode'])

    #### Lastly, we wrap these graph with `AllenNlpGraph` functionalities to get the full learning based program.
    lbp = AllenNlpGraph(graph, **config.graph)
    return lbp


#### The main entrance of the program.
def main():
    save_config = Config.deepclone()
    #### 1. "Knowledge Declaration" to get a graph, as a partial program.
    graph = knowledge_declaration()

    #### 2. "Data Declaration" and "Learning Declaration" to connect sensors and learners and get the full program.
    lbp = data_and_learning_declaration(graph, Config.Model)

    #### 3. Train and save the model
    #### To have better reproducibility, we initial the random seeds of all subsystems.
    seed()
    #### Train the model with inference functionality inside.
    lbp.train(Config.Data, Config.Train)
    #### Save the model, including vocabulary use to index the tokens.
    save_to = Config.Train.trainer.serialization_dir or '/tmp/emr'
    lbp.save(save_to, config=save_config)

####
"""
This example show a full pipeline how to work with `regr`.
"""

if __name__ == '__main__':
    main()
