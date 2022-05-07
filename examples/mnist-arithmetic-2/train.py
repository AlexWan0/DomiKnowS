import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

from data import get_readers
from functools import partial
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

from model import build_program
import config

trainloader, validloader, testloader = get_readers()

#program = LearningBasedProgram(graph, Model)

def get_classification_report(program, reader, total=None, verbose=False):
    digit_pred_a = []
    sum_pred_a = []

    digit_label_a = []
    sum_label_a = []

    for i, node in tqdm(enumerate(program.populate(reader, device='auto')), total=total):
        node.inferILPResults()

        #print(node.getAttributes())

        suffix = '/ILP'

        digit0_pred = torch.argmax(node.getAttribute(f'<digits0>{suffix}'))
        digit1_pred = torch.argmax(node.getAttribute(f'<digits1>{suffix}'))
        summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))

        if verbose:
            print(f"PRED: {digit0_pred} + {digit1_pred}")

        digit0_label = node.getAttribute('digit0_label').item()
        digit1_label = node.getAttribute('digit1_label').item()
        summation_label = node.getAttribute('<summations>/label').item()

        if verbose:
            print(f"LABEL: {digit0_label} + {digit1_label}")

        digit_pred_a.append(digit0_pred)
        digit_pred_a.append(digit1_pred)

        digit_label_a.append(digit0_label)
        digit_label_a.append(digit1_label)

        sum_pred_a.append(summation_pred)
        sum_label_a.append(summation_label)

    print(classification_report(digit_label_a, digit_pred_a))
    print(classification_report(sum_label_a, sum_pred_a))


program = build_program()

#get_classification_report(program, validloader, total=config.num_valid, verbose=False)

for i in range(1, 11):
    print("EPOCH", i)

    program.train(trainloader,
              train_epoch_num=1,
              Optim=lambda x: torch.optim.Adam(x, lr=0.001),
              device='auto')

    # validation
    if i % 3 == 0:
        get_classification_report(program, validloader, total=config.num_valid, verbose=False)

get_classification_report(program, validloader, total=config.num_valid, verbose=False)
