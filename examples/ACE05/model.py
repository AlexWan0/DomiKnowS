import torch

from regr.program import POIProgram
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.pytorch import PoiModel
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor, FunctionalReaderSensor, cache, TorchCache, JointSensor
from regr.sensor.pytorch.tokenizers.transformers import TokenizerEdgeSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CandidateSensor, CandidateRelationSensor, CandidateEqualSensor, CompositionCandidateSensor

from sensors.readerSensor import MultiLevelReaderSensor, SpanLabelSensor, CustomMultiLevelReaderSensor, LabelConstantSensor
from models import Tokenizer, BERT, SpanClassifier, token_to_span_candidate_emb, span_to_pair_emb, find_is_a, find_event_arg, token_to_span_label, makeSpanPairs, makeSpanAnchorPairs


def model(graph):
    from ace05.graph import graph, entities_graph, events_graph
    from ace05.graph import document, token, span_candidate, span, span_annotation, anchor_annotation, event, pair

    # document
    document['text'] = ReaderSensor(keyword='text')

    # document -> token
    document_contains_token = document.relate_to(token)[0]
    token[document_contains_token.forward, 'ids', 'offset'] = cache(JointSensor)(document['text'], forward=Tokenizer(), cache=TorchCache(path="./cache/tokenizer"))
    token['emb'] = ModuleLearner('ids', module=BERT())

    # span annotation
    span_annotation['index'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.text")
    span_annotation['start'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.start")
    span_annotation['end'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.end")
    span_annotation['type'] = CustomMultiLevelReaderSensor(keyword="spans.*.type")
    span_annotation['subtype'] = CustomMultiLevelReaderSensor(keyword="spans.*.subtype")
    
    anchor_annotation['index'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.text")
    anchor_annotation['start'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.start")
    anchor_annotation['end'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.end")
    anchor_annotation['type'] = CustomMultiLevelReaderSensor(keyword="events.*.type")
    anchor_annotation['subtype'] = CustomMultiLevelReaderSensor(keyword="events.*.subtype")

    # token -> span and span equality extention
    span_contains_token = span.relate_to(token)[0]
    span_equal_annotation = span.relate_to(span_annotation)[0]
    anchor_equal_annotation = span.relate_to(anchor_annotation)[0]

    def token_to_span_fn(token_offset, sanno_start, sanno_end, aanno_start, aanno_end):
        num_token = token_offset.shape[0]
        token_start = token_offset[:,0]
        token_end = token_offset[:,1]
        spans = []
        # dummy spans
        for start in range(0, num_token, 4):
            spans.append((start, start+2))
        sannos = []
        for start, end in zip(sanno_start, sanno_end):
            start_token = torch.nonzero(torch.logical_and(token_start <= start, start < token_end), as_tuple=False)[0, 0]
            end_token = torch.nonzero(torch.logical_and(token_start < end, end <= token_end), as_tuple=False)[0, 0]
            try:
                span_index = spans.index((start_token, end_token))
            except ValueError:
                span_index = len(spans)
                spans.append((start_token, end_token))
            sannos.append(span_index)
        aannos = []
        for start, end in zip(aanno_start, aanno_end):
            start_token = torch.nonzero(torch.logical_and(token_start <= start, start < token_end), as_tuple=False)[0, 0]
            end_token = torch.nonzero(torch.logical_and(token_start < end, end <= token_end), as_tuple=False)[0, 0]
            try:
                span_index = spans.index((start_token, end_token))
            except ValueError:
                span_index = len(spans)
                spans.append((start_token, end_token))
            aannos.append(span_index)

        token_mapping = torch.zeros(len(spans), num_token)
        for j, (start, end) in enumerate(spans):
            token_mapping[j, start:end] = 1
        sanno_mapping = torch.zeros(len(spans), len(sanno_start))
        for i, index in enumerate(sannos):
            sanno_mapping[index, i] = 1
        aanno_mapping = torch.zeros(len(spans), len(aanno_start))
        for i, index in enumerate(aannos):
            aanno_mapping[index, i] = 1
        return token_mapping, sanno_mapping, aanno_mapping
    span[span_contains_token.backward, span_equal_annotation.backward, anchor_equal_annotation.backward] = JointSensor(token['offset'], span_annotation['start'], span_annotation['end'], anchor_annotation['start'], anchor_annotation['end'], forward=token_to_span_fn)
    span['emb'] = FunctionalSensor(span_contains_token.backward('emb'), forward=lambda x: x)

    # span -> base types
    for concept in find_is_a(entities_graph, span):
        print(f'Creating learner/reader for span -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        # span_annotation[concept] = LabelConstantSensor(concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

    # entity -> major classes
    entity = entities_graph['Entity']
    for concept in find_is_a(entities_graph, entity):
        if '.' in concept.name:  # skip 'Class.', 'Role.', etc.
            continue
        print(f'Creating learner/reader for entity -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # entity -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            if '.' in sub_concept.name:  # skip 'Class.', 'Role.', etc.
                continue
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # value -> major classes
    value = entities_graph['value']
    for concept in find_is_a(entities_graph, value):
        print(f'Creating learner/reader for value -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # value -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # span -> pair
    arg1, arg2 = pair.relate_to(span)
    pair[arg1.backward, arg2.backward] = CompositionCandidateSensor(
        span['emb'],
        relations=(arg1.backward, arg2.backward),
        forward=lambda *_: True)
    # pair['index'] = CandidateSensor(span['emb'], forward=lambda *_: True)
    pair['emb'] = FunctionalSensor(span['emb'], forward=span_to_pair_emb)

    # event
    for concept in find_is_a(events_graph, event):
        print(f'Creating learner/reader for event -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        # span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # event sub classes
        for sub_concept in find_is_a(events_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            # span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

            # all event argument rules are associated with event subtypes
            # pair -> event argument
            for event_arg in find_event_arg(events_graph, sub_concept):
                print(f'Creating learner/reader for pair -> {sub_concept.name}\'s {event_arg.name}')
                pair[event_arg] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
                # pair[event_arg] = ?

    # program = POIProgram(graph, poi=(span, pair))
    program = PrimalDualProgram(graph, PoiModel, poi=(document, token, span, pair))

    return program
