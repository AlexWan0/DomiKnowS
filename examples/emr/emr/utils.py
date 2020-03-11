def seed(s=1, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ['PYTHONHASHSEED'] = str(s)  # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)  # this function will call torch.cuda.manual_seed_all(s) also

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Namespace(dict):
    def __init__(self, __dict={}, **kwargs):
        self.__dict__ = self
        self.update(__dict)
        self.update(kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Namespace(v)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ','.join('\'{}\':{}'.format(k,v) for k,v in self.items()))

    def clone(self):
        from copy import copy
        return Namespace(copy(self))

    def deepclone(self):
        from copy import deepcopy
        return Namespace(deepcopy(self))

def caller_source():
    import inspect

    for frame in inspect.getouterframes(inspect.currentframe(), context=1)[2:]:
        if frame.code_context is not None:
            try:
                with open(frame.filename, 'r') as fin:
                    return fin.read()
                break
            except FileNotFoundError as ex:
                ex = type(ex)('{}\n'
                              'Please run from a file base environment, '
                              'rather than something like notebook.'.format(ex))
                raise ex
    else:
        raise RuntimeError('Who is calling?')
