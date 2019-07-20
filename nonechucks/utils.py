import collections

from itertools import chain
from functools import wraps

import torch
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate
from torch._six import string_classes


class transparent_partial(object):
    def __init__(self, func, *args, **kw):
        self.__dict__['__func'] = func
        self.__dict__['__args'] = args
        self.__dict__['__kw'] = kw

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.__dict__['__func'], item)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        return setattr(self.__dict__['__func'], key, value)

    def __call__(self, *args, **kw):
        return self.__dict__['__func'](*self.__dict__['__args'], *args,
                                       **self.__dict__['__kw'], **kw)


class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.memoize = True

    def __get__(self, instance, owner):
        if instance is None:
            return self.func
        return transparent_partial(self, instance)

    def __call__(self, instance, *args, **kw):
        if not self.memoize:
            return self.func(instance, *args, **kw)

        key = (instance, args, frozenset(kw.items()))
        try:
            res = self.cache[key]
        except KeyError:
            res = self.cache[key] = self.func(instance, *args, **kw)
        return res


def collate_batches(batches, collate_fn=default_collate):
    """Collate multiple batches."""
    error_msg = "batches must be tensors, dicts, or lists; found {}"
    if isinstance(batches[0], torch.Tensor):
        return torch.cat(batches, 0)
    elif isinstance(batches[0], collections.Sequence):
        return list(chain(*batches))
    elif isinstance(batches[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batches])
                for key in batches[0]}
    raise TypeError((error_msg.format(type(batches[0]))))


def batch_len(batch):
    # error_msg = "batch must be tensor, dict, or list: found {}"
    if isinstance(batch, list):
        if isinstance(batch[0], string_classes):
            return len(batch)
        else:
            return len(batch[0])
    elif isinstance(batch, collections.Mapping):
        first_key = list(batch.keys())[0]
        return len(batch[first_key])
    return len(batch)


def slice_batch(batch, start=None, end=None):
    if isinstance(batch, list):
        if isinstance(batch[0], string_classes):
            return batch[start:end]
        else:
            return [sample[start:end] for sample in batch]
    elif isinstance(batch[0], collections.Mapping):
        return {key: batch[key][start:end] for key in batch}
    else:
        return batch[start:end]
