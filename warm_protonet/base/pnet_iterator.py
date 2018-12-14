from typing import Iterable, Dict, Iterator, Optional, List
import logging
import math
import random

import numpy
from overrides import overrides

from allennlp.common import Params
from allennlp.data.iterators.data_iterator import DataIterator

import logging
from typing import Dict, Union, Iterable, Iterator, List, Optional, Tuple
import itertools
import math
import random

import torch

from allennlp.common.util import is_lazy, lazy_groups_of, ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def add_epoch_number(batch: Batch, epoch: int) -> Batch:
    """
    Add the epoch number to the batch instances as a MetadataField.
    """
    for instance in batch.instances:
        instance.fields['epoch_num'] = MetadataField(epoch)
    return batch


@DataIterator.register("pnet")
class PnetIterator(DataIterator):
    """
    This iterators differs from basic iterator in one thing: we don't shuffle objects to make batches.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : int, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    """

    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        super().__init__(batch_size, instances_per_epoch, max_instances_in_memory)
        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch
        self._max_instances_in_memory = max_instances_in_memory

        self._cursors: Dict[int, Iterator[Instance]] = {}

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1) -> Iterator[TensorDict]:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.

        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        cuda_device : ``int``
            If cuda_device >= 0, GPUs are available and Pytorch was compiled with CUDA support, the
            tensor will be copied to the cuda_device specified.
        """
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            self._epochs[key] = epoch

            batches = self._create_batches(instances, shuffle)
            batches = [b for b in batches]
            if len(batches) > 1000:
                batches = numpy.random.choice(batches, size=300, replace=False)
            # Should we add the instances to the cache this epoch?
            add_to_cache = self._cache_instances and key not in self._cache

            for batch in batches:
                if self._track_epoch:
                    add_epoch_number(batch, epoch)

                if self.vocab is not None:
                    batch.index_instances(self.vocab)

                padding_lengths = batch.get_padding_lengths()
                logger.debug("Batch padding lengths: %s", str(padding_lengths))
                logger.debug("Batch size: %d", len(batch.instances))
                tensor_dict = batch.as_tensor_dict(padding_lengths,
                                                   cuda_device=cuda_device)

                if add_to_cache:
                    self._cache[key].append(tensor_dict)

                yield tensor_dict

    @overrides
    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        if is_lazy(instances) and self._instances_per_epoch is None:
            # Unable to compute num batches, so just return 1.
            return 1
        elif self._instances_per_epoch is not None:
            return math.ceil(self._instances_per_epoch / self._batch_size)
        else:
            # Not lazy, so can compute the list length.
            return math.ceil(len(ensure_list(instances)) / self._batch_size)

    def _take_instances(self,
                        instances: Iterable[Instance],
                        max_instances: Optional[int] = None) -> Iterator[Instance]:
        """
        Take the next `max_instances` instances from the given dataset.
        If `max_instances` is `None`, then just take all instances from the dataset.
        If `max_instances` is not `None`, each call resumes where the previous one
        left off, and when you get to the end of the dataset you start again from the beginning.
        """
        # If max_instances isn't specified, just iterate once over the whole dataset
        if max_instances is None:
            yield from iter(instances)
        else:
            # If we don't have a cursor for this dataset, create one. We use ``id()``
            # for the key because ``instances`` could be a list, which can't be used as a key.
            key = id(instances)
            iterator = self._cursors.get(key, iter(instances))

            while max_instances > 0:
                try:
                    # If there are instances left on this iterator,
                    # yield one and decrement max_instances.
                    yield next(iterator)
                    max_instances -= 1
                except StopIteration:
                    # None left, so start over again at the beginning of the dataset.
                    iterator = iter(instances)

            # We may have a new iterator, so update the cursor.
            self._cursors[key] = iterator

    def _memory_sized_lists(self, instances: Iterable[Instance]) -> Iterable[List[Instance]]:
        """
        Breaks the dataset into "memory-sized" lists of instances,
        which it yields up one at a time until it gets through a full epoch.

        For example, if the dataset is already an in-memory list, and each epoch
        represents one pass through the dataset, it just yields back the dataset.
        Whereas if the dataset is lazily read from disk and we've specified to
        load 1000 instances at a time, then it yields lists of 1000 instances each.
        """
        lazy = is_lazy(instances)

        # Get an iterator over the next epoch worth of instances.
        iterator = self._take_instances(instances, self._instances_per_epoch)

        # We have four different cases to deal with:

        # With lazy instances and no guidance about how many to load into memory,
        # we just load ``batch_size`` instances at a time:
        if lazy and self._max_instances_in_memory is None:
            yield from lazy_groups_of(iterator, self._batch_size)
        # If we specified max instances in memory, lazy or not, we just
        # load ``max_instances_in_memory`` instances at a time:
        elif self._max_instances_in_memory is not None:
            yield from lazy_groups_of(iterator, self._max_instances_in_memory)
        # If we have non-lazy instances, and we want all instances each epoch,
        # then we just yield back the list of instances:
        elif self._instances_per_epoch is None:
            yield ensure_list(instances)
        # In the final case we have non-lazy instances, we want a specific number
        # of instances each epoch, and we didn't specify how to many instances to load
        # into memory. So we convert the whole iterator to a list:
        else:
            yield list(iterator)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        As you can see, we don't shuffle our objects here.
        """
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            iterator = iter(instance_list)
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                yield Batch(batch_instances)

    @classmethod
    def from_params(cls, params: Params) -> 'PnetIterator':
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
