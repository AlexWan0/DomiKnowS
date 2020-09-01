from itertools import chain

import torch
from transformers import BertTokenizer, BertTokenizerFast, BertModel


def cartesian_concat(*inputs):
    # torch cat is not broadcasting, do repeat manually
    input_iter = iter(inputs)
    output = next(input_iter)
    for input in input_iter:
        *dol, dof = output.shape
        *dil, dif = input.shape
        output = output.view(*dol, *(1,)*len(dil), dof).repeat(*(1,)*len(dol), *dil, 1)
        input = input.view(*(1,)*len(dol), *dil, dif).repeat(*dol, *(1,)*len(dil), 1)
        output = torch.cat((output, input), dim=-1)
    return output

TRANSFORMER_MODEL = 'bert-base-uncased'

def Tokenizer():
    return BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

# emb_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
# to freeze BERT, uncomment the following
# for param in emb_model.base_model.parameters():
#     param.requires_grad = False
class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        # to freeze BERT, uncomment the following
        # for param in emb_model.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, input):
        input = input.unsqueeze(0)
        out, *_ = self.module(input)
        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out

def token_to_span_candidate(spans, start, end):
    length = end.getAttribute('index') - start.getAttribute('index')
    if length > 0 and length < 10:
        return True
    else:
        return False

def span_candidate_emb(token_emb, span_candidate_index):
    embs = cartesian_concat(token_emb, token_emb)
    span_candidate_index = span_candidate_index.rename(None)
    span_candidate_index = span_candidate_index.unsqueeze(-1).repeat(1, 1, embs.shape[-1])
    selected = embs.masked_select(span_candidate_index).view(-1, embs.shape[-1])
    return selected

def span_label(span_index, token_offset, data):
    span_index = span_index.rename(None)
    span_label = span_index.clone()
    for i, j in span_index.nonzero():
        for mention in chain(*map(lambda span: span.mentions, data)):
            if mention.start == i and mention.end == j:
                break
        else:  # match not found
            span_label[i, j] = 0
    selected = span_label.masked_select(span_index).view(-1)
    return selected

def span_emb(span_candidate_emb, span_index):
    span_index = span_index.rename(None)
    span_index = span_index.unsqueeze(-1).repeat(1, span_candidate_emb.shape[-1])
    selected = span_candidate_emb.masked_select(span_index).view(-1, span_candidate_emb.shape[-1])
    return selected


def find_is_a(graph, base_concept):
    for name, concept in graph.concepts.items():
        if base_concept in map(lambda rel: rel.dst, concept.is_a()):
            yield concept