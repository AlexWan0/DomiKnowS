import torch
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from models import *
from utils import *
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor


def program_declaration(cur_device, *, PMD=False, beta=0.5, sampleloss=False, sampleSize=1):
    from graph import graph, paragraph, paragraph_contain, event_relation, \
        relation_classes, symmetric, s_event1, s_event2, transitive, t_event1, t_event2, t_event3

    # Reading directly from data table
    paragraph["files"] = ReaderSensor(keyword="files", device=cur_device)
    paragraph["eiids1"] = ReaderSensor(keyword="eiids1", device=cur_device)
    paragraph["eiids2"] = ReaderSensor(keyword="eiids2", device=cur_device)
    paragraph["x_sent_list"] = ReaderSensor(keyword="x_sent_list", device=cur_device)
    paragraph["y_sent_list"] = ReaderSensor(keyword="y_sent_list", device=cur_device)
    paragraph["x_position_list"] = ReaderSensor(keyword="x_position_list", device=cur_device)
    paragraph["y_position_list"] = ReaderSensor(keyword="y_position_list", device=cur_device)
    paragraph["x_event_list"] = ReaderSensor(keyword="x_event_list", device=cur_device)
    paragraph["y_event_list"] = ReaderSensor(keyword="y_event_list", device=cur_device)
    paragraph["relation_list"] = ReaderSensor(keyword="relation_list", device=cur_device)

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def relation_str_to_list(relations):
        rel = []
        flags = []  # 0 for temporal, otherwise 1
        rel_index = {"BEFORE": '4', "AFTER": '5', "EQUAL": '6', "VAGUE": '7',
                     "SuperSub": '0', "SubSuper": '1', "Coref": '2', "NoRel": '3'}
        # rel_index = {"BEFORE": '0', "AFTER": '1', "EQUAL": '2', "VAGUE": '3'}
        for relation in relations:
            rel += [rel_index[relation]]
            flags.append(0 if int(rel_index[relation]) < 4 else 1)
        return str_to_int_list(rel), str_to_int_list(flags)

    def str_to_token_list(x):
        tokens_list = x.split("@@")
        return torch.IntTensor([[int(i) for i in eval(tokens)] for tokens in tokens_list]).to(cur_device)

    def make_event(files, eiids1, eiids2, x_sent_list, y_sent_list,
                   x_position_list, y_position_list, x_event_list, y_event_list, relation_list):
        # Seperate them from batch to seperate dataset
        # Note that x_tokens_list need to use split -> eval -> torch.tensor
        eiid1_list = str_to_int_list(eiids1.split("@@"))
        eiid2_list = str_to_int_list(eiids2.split("@@"))
        x_sent = str_to_token_list(x_sent_list)
        y_sent = str_to_token_list(y_sent_list)
        x_pos_list = str_to_int_list(x_position_list.split("@@"))
        y_pos_list = str_to_int_list(y_position_list.split("@@"))
        rel, flags = relation_str_to_list(relation_list.split("@@"))
        return torch.ones(len(files.split("@@")), 1), files.split("@@"), \
               eiid1_list, eiid2_list, x_sent, y_sent, x_pos_list, y_pos_list, x_event_list.split("@@"), \
               y_event_list.split("@@"), rel, flags

    event_relation[paragraph_contain,
                   "file", "eiid1", "eiid2", "x_sent", "y_sent", "x_pos", "y_pos", "x_event", "y_event", "rel_", "flags"] = \
        JointSensor(paragraph["files"], paragraph["eiids1"], paragraph["eiids2"],
                    paragraph["x_sent_list"], paragraph["y_sent_list"],
                    paragraph["x_position_list"], paragraph["y_position_list"],
                    paragraph["x_event_list"], paragraph["y_event_list"],
                    paragraph["relation_list"], forward=make_event, device=cur_device)

    def label_reader(_, label):
        return label

    event_relation[relation_classes] = FunctionalSensor(paragraph_contain, "rel_",
                                                        forward=label_reader, label=True, device=cur_device)

    # BiLSTM setting
    hidden_layer = 256
    roberta_size = 'roberta-base'
    out_model = BiLSTM(768 if roberta_size == 'roberta-base' else 1024,
                       hidden_layer, num_layers=1, roberta_size=roberta_size)
    # out_model = Robert_Model()
    event_relation["x_output"] = ModuleLearner("x_sent", "x_pos", module=out_model, device=cur_device)
    event_relation["y_output"] = ModuleLearner("y_sent", "y_pos", module=out_model, device=cur_device)


    emb_path = "common_sense/common_sense.txt"
    mdl_path = "common_sense/pairwise_model_0.3_200_1.pt"
    ratio = 0.3
    layer = 1
    emb_size = 256
    granularity = 0.05
    bigramStats_dim = 2
    common_sense_model = common_sense_from_NN(emb_path, mdl_path, ratio, layer, emb_size, cur_device=cur_device)
    common_sense_EMB = nn.Embedding(int(1.0/granularity) * bigramStats_dim, emb_size)

    def common_sense_emb(_, verbs1, verbs2):
        common_sense_embs = []
        for ind, v1 in enumerate(verbs1):
            v2 = verbs2[ind]
            bigramstats = common_sense_model.getCommonSense(v1, v2)
            common_sense_emb = common_sense_EMB(torch.LongTensor(
                [min(int(1.0 / granularity) - 1, int(bigramstats[0][0] / granularity))])).view(1, -1)
            for i in range(1, bigramStats_dim):
                tmp = common_sense_EMB(torch.LongTensor([(i - 1) * int(1.0 / granularity) + min(
                    int(1.0 / granularity) - 1, int(bigramstats[0][i] / granularity))])).view(1, -1)
                common_sense_emb = torch.cat((common_sense_emb, tmp), 1)
            common_sense_embs.append(common_sense_emb.tolist()[0])
        return torch.Tensor(common_sense_embs)

    def make_MLP_input(_, x, y, common_sense):
        subXY = torch.sub(x, y)
        mulXY = torch.mul(x, y)
        return_input = torch.cat((x, y, subXY, mulXY, common_sense), 1)
        return return_input

    event_relation["common_sense"] = FunctionalSensor(paragraph_contain, "x_event", "y_event",
                                                      forward=common_sense_emb, device=cur_device)
    event_relation["MLP_input"] = FunctionalSensor(paragraph_contain, "x_output", "y_output", "common_sense",
                                                   forward=make_MLP_input, device=cur_device)

    event_relation[relation_classes] = ModuleLearner("MLP_input", module=BiLSTM_MLP(out_model.last_layer_size, 512, 8),
                                                     device=cur_device)

    from regr.program.metric import PRF1Tracker, PRF1Tracker, DatanodeCMMetric, MacroAverageTracker
    from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
    from regr.program import LearningBasedProgram, SolverPOIProgram
    from regr.program.lossprogram import SampleLossProgram, PrimalDualProgram
    from regr.program.model.pytorch import model_helper, PoiModel, SolverModel

    # Define the same weight as original paper
    HierPC = 1802.0
    HierCP = 1846.0
    HierCo = 758.0
    HierNo = 63755.0
    HierTo = HierPC + HierCP + HierCo + HierNo  # total number of event pairs
    weights = torch.FloatTensor([0.25 * HierTo / HierPC, 0.25 * HierTo / HierCP,
                                 0.25 * HierTo / HierCo, 0.25 * HierTo / HierNo,
                                 0.25 * 818.0 / 412.0, 0.25 * 818.0 / 263.0,
                                 0.25 * 818.0 / 30.0, 0.25 * 818.0 / 113.0]).to(cur_device)
    # weights = torch.FloatTensor([0.25 * 818.0 / 412.0, 0.25 * 818.0 / 263.0,
    #                              0.25 * 818.0 / 30.0, 0.25 * 818.0 / 113.0]).to(cur_device)

    # Initial program using only ILP
    symmetric[s_event1.reversed, s_event2.reversed] = CompositionCandidateSensor(
        relations=(s_event1.reversed, s_event2.reversed),
        forward=check_symmetric, device=cur_device)

    transitive[t_event1.reversed, t_event2.reversed, t_event3.reversed] = CompositionCandidateSensor(
        relations=(t_event1.reversed, t_event2.reversed, t_event3.reversed),
        forward=check_transitive, device=cur_device)

    inferList = ['ILP', 'local/argmax']  # ['ILP', 'local/argmax']
    poi_list = [event_relation, relation_classes, symmetric, transitive]
    if PMD:
        program = PrimalDualProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=inferList,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss(weight=weights)),
                                    beta=beta,
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif sampleloss:
        program = SampleLossProgram(graph, SolverModel, poi=poi_list,
                                    inferTypes=inferList,
                                    loss=MacroAverageTracker(NBCrossEntropyLoss(weight=weights)),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    sample=True,
                                    sampleSize=sampleSize,
                                    sampleGlobalLoss=True)
    else:
        program = SolverPOIProgram(graph,
                                   poi=poi_list,
                                   inferTypes=inferList,
                                   loss=MacroAverageTracker(NBCrossEntropyLoss(weight=weights)),
                                   metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                           'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program
