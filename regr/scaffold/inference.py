from .. import Graph
from typing import Dict
from torch import Tensor
import torch
import pandas as pd


DataInstance = Dict[str, Tensor]


def inference(
    graph: Graph,
    data: DataInstance
) -> DataInstance:
    groups = [  # TODO: replace by constraint in graph, or group discover, later
        ['people', 'organization', 'location', 'other', 'o'],
        ['work_for', 'located_in', 'live_in', 'orgbase_on'],
    ]

    # table columns, as many table columns as groups
    tables = [[] for _ in groups]
    wild = []  # for those not in any group
    # for each subgraph.concept[prop] that has multiple assignment
    for subgraph, concept, prop, module_funcs in graph.get_multiassign():
        # find the group it goes to
        # TODO: update later with group discover?
        for group, table in zip(groups, tables):
            if concept.name in group:
                # for each assignment, [0] for label, [1] for pred
                for i, ((module, func), conf) in enumerate(module_funcs):
                    # consider only prediction here
                    # TODO: use a non-label aggregated prediction
                    if i == 1:
                        # add the concept (might useful) and function handle to the table (column)
                        table.append((concept, prop, func))
                        break  # TODO: if aggregated, no need to break
        else:  # for group, table
            # belongs to no group
            # still do something, differently
            wild.append((concept, prop, func))

    # now we have (batch, ) in predictions, but inference may process one item at a time
    # should organize in this way (batch, len, ..., t/f, column), then we can iter through batches
    # note that, t/f is a redundant dim, that gives 2 scalars: [1-p, p], maybe needed to squeeze
    # using the pindex and index_select requires the above facts
    valuetables = []
    batch_size = None
    for table in tables:
        # assume all of them give same dims of output: (batch, len, ..., t/f)
        values = []
        for column in table:
            concept, prop, func = column
            value = func(data)  # (batch, len, ..., t/f)
            # at t/f dim, 0 for 1-p, 1 for p
            pindex = torch.tensor(1, device=value.device).long()
            # (batch, len, ..., 1) # need tailing 1 for cat
            value = value.index_select(-1, pindex)
            # get/check the batch_size
            if batch_size is None:
                batch_size = value.size()[0]
            else:
                assert batch_size == value.size()[0]
            values.append(value)
        values = torch.cat(values, dim=-1)  # (batch, len, ..., ncls)
        # then it has the same order as tables, where we can refer to related concept
        valuetables.append(values)
    # we need all columns to be placed, for all tables, before inference
    # now we have

    updated_valuetables_batch = [[] for _ in valuetables]
    # for each batch
    for batch_index in torch.arange(batch_size, device=values.device):
        inference_tables = []
        for values, table in zip(valuetables, tables):
            # use only current batch
            # 0 for batch, resulting (len, ..., ncls)
            values = values.index_select(0, batch_index)
            values = values.squeeze(dim=0)
            names, props = zip(*[(concept.name, prop)
                                 for concept, prop, _ in table])
            # now values is the table we need
            # and names is the list of grouped concepts (name of the columns)
            inference_tables.append((names, props, values))

        # data structure convert
        phrase = None  # TODO: since it not using now. if it is needed later, will pass it somewhere else

        phrasetable = inference_tables[0][2].clone().cpu().detach().numpy()
        graphResultsForPhraseToken = pd.DataFrame(
            phrasetable,
            index=[str(i) for i in range(inference_tables[0][2].size()[0])],
            columns=[concept.name for concept, prop, _ in tables[0]])

        graphtable = inference_tables[1][2].clone().cpu().detach().numpy()
        graphResultsForPhraseRelation = dict()
        for i, (composed_concept, _, _) in enumerate(tables[1]):
            # each relation
            graphResultsForPhraseRelation[composed_concept.name] = pd.DataFrame(
                graphtable[:, :, i],
                index=[str(i)
                       for i in range(inference_tables[0][2].size()[0])],
                columns=[str(i) for i in range(inference_tables[0][2].size()[0])])

        # do inference
        from ..ilpSelectClassification import calculateIPLSelection
        iplResults = calculateIPLSelection(
            phrase, graph, graphResultsForPhraseToken, graphResultsForPhraseRelation)
        # iplResults is a dictionary of {token: conceptName}

        # convert back
        for i, (updated_batch, (names, props, values)) in enumerate(zip(updated_valuetables_batch, inference_tables)):
            # values: tensor (len, ..., ncls)
            # updated_batch: list of batches of result of tensor (len, ..., ncls)
            #updated = torch.zeros(values.size())
            # do something to query iplResults to fill updated
            #
            # implement below
            #
            if i == 0:
                updated = torch.tensor(
                    iplResults.to_numpy(), device=values.device).float()
            elif i == 1:
                # skip compose since it is not return for now
                continue
            else:
                # should be nothing here
                pass
            #
            # implement above
            #

            # add to updated batch
            updated_batch.append(updated)
    # updated_valuetables_batch is List(tables)[List(batch_size)[Tensor(len, ..., ncls)]]

    # put it back into one piece
    # we want List(tables)[List(ncls)[Tensor(batch, len, ..., 2)]]
    # be careful of that the 2 need extra manuplication
    for updated_batch, (names, props, values) in zip(updated_valuetables_batch, inference_tables):
        # updated_batch: List(batch_size)[Tensor(len, ..., ncls)]
        # List(batch_size)[Tensor(1, len, ..., ncls)]
        updated_batch = [updated.unsqueeze(dim=0) for updated in updated_batch]
        # Tensor(batch, len, ..., ncls)
        updated_batch_tensor = torch.cat(updated_batch, dim=0)
        # for each class in ncls
        size = updated_batch_tensor.size()

        for icls, name, prop in zip(torch.arange(size[-1], device=updated_batch_tensor.device), names, props):
            value = updated_batch_tensor.index_select(
                0, icls)  # Tensor(batch, len, ...,)
            value = value.unsqueeze(dim=-1)  # Tensor(batch, len, ..., 1)
            # Tensor(batch, len, ..., 2)
            value = torch.cat([1 - value, value], dim=-1)
            fullname = '{}[{}]-{}'.format(graph[name].fullname, prop, 1)
            # put it back finally
            data[fullname] = value

    return data
