from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer, \
    ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    line = line.strip()
    return not line or line == """-DOCSTART- -X- -X- O"""


_VALID_LABELS = {'ner', 'pos', 'chunk'}

import os
import json
from glob import glob
import ftplib
import numpy as np
import random
import urllib.request
import tarfile
import re
import pandas as pd
import copy

"""
This file is a changed analogous file from allennlp. That's why it may contain irrelevant comments.
"""

def tokenize(s):
    return re.findall(r"[\w']+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)


def snips_reader(file='train', dataset_download_path='../ontonotes/', valid_class=None, random_seed=None, drop_empty=False,
                 valid_part=None):
    sentences = []
    ys = []
    if file == 'test.txt':
        file_name = 'valid.txt' # We don't use test.txt. We don't have a validation procedure so we test on "valid.txt".
    else:
        file_name = file
    with open(dataset_download_path + file_name, "r") as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                fields = [line.strip().split() for line in lines]
                tokens, ner_tags = [list(field) for field in zip(*fields)]
                sentences.append([[token, tag] for token, tag in zip(tokens, ner_tags)])
                ys += ner_tags

    np.random.seed(random_seed)
    np.random.shuffle(sentences)
    # Number of sentences contains target class we used to train
    train_sentences = 20
    # Number of sentences contains target class we used to train
    train_train_sentences = 40
    n_episodes = 10000
    total_sentences = 100

    super_list = []

    if file == 'valid.txt' or file == 'test.txt':
        # Erase all classes except target class
        for sent in sentences:
            for word in sent:
                if word[1][2:] != valid_class:
                    word[1] = 'O'
        validation_all = []
        validation_batch = []
        empty_valid = []
        number_appeared = 0

        # Split all sentences into 3 groups: 20 sentences we are going to train, the rest of sentences that contain
        # target class and all the empty sentences
        for sentence in sentences:
            ys_here = [xy[1] for xy in sentence]
            if np.unique(ys_here).shape[0] > 1 and len(validation_batch) < train_sentences:
                validation_batch.append(sentence)
                number_appeared += 1
            elif np.unique(ys_here).shape[0] > 1:
                number_appeared += 1
                validation_all.append(sentence)
            elif np.unique(ys_here).shape[0] == 1:
                empty_valid.append(sentence)

        # Here we add some number of empty sentences to train dataset to balance the classes
        add_more = int((train_sentences / 2) * (len(sentences) / number_appeared - 1))
        add_more = max(50, add_more)
        # We don't use this sentences in the test because we don't need them to compute prototypes

        train = validation_batch

        # To be honest, I can't remember why I did this
        if (len(validation_all) + len(empty_valid[add_more:])) % (total_sentences - train_sentences) == 1:
            empty_valid = empty_valid[:-1]

        # create final test set
        test = validation_all + empty_valid[add_more:]

        # Generate batches: 20 sentences that we use to compute prototypes and 80 sentences to predict.
        # We use the same 20 sentences and vary 80 sentences among all the test sentences. Then we concatenate this.
        # Note: we will not shuffle our data inside iterator.
        query_number = total_sentences - train_sentences
        for i in range(int(np.ceil(len(test) / query_number))):
            super_list += copy.deepcopy(
                train + test[(i * query_number):(min(len(test) - 1, (i + 1) * query_number))])

    if file == 'train.txt':
        valid_sentences = []
        with open(dataset_download_path + 'valid.txt', "r") as data_file:
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    tokens, ner_tags = [list(field) for field in zip(*fields)]
                    valid_sentences.append([[token, tag] for token, tag in zip(tokens, ner_tags)])
        np.random.seed(random_seed)
        np.random.shuffle(valid_sentences)
        # Erase all classes except target class
        for sent in valid_sentences:
            for word in sent:
                if word[1][2:] != valid_class:
                    word[1] = 'O'

        # Split all sentences into 2 groups: 20 sentences we are going to train and all the empty sentences
        validation_batch = []
        empty_valid = []
        number_appeared = 0
        for sentence in valid_sentences:
            ys_here = [xy[1] for xy in sentence]
            if np.unique(ys_here).shape[0] > 1 and len(validation_batch) < train_sentences:
                validation_batch.append(sentence)
                number_appeared += 1
            elif np.unique(ys_here).shape[0] > 1:
                number_appeared += 1
            elif np.unique(ys_here).shape[0] == 1:
                empty_valid.append(sentence)
        add_more = int((train_sentences / 2) * (len(valid_sentences) / number_appeared - 1))
        add_more = max(50, add_more)
        # We use a certain number of empty sentences to train. This number depends on the balance of classes.
        empty_valid = empty_valid[:add_more]

        # We are going to sample query and support sentences from different groups.
        support_sentences = sentences[:(len(sentences) // 2)]
        query_sentences = sentences[(len(sentences) // 2):]
        # Erase the target class
        for sent in support_sentences:
            for word in sent:
                if word[1][2:] == valid_class:
                    word[1] = 'O'

        for sent in query_sentences:
            for word in sent:
                if word[1][2:] == valid_class:
                    word[1] = 'O'

        # Here we create many datasets. In every dataset we erase all the classes except the chosen one.
        ys_sup, counts = np.unique(np.concatenate([[xy[1][2:] for xy in sentence] for sentence in sentences]),
                                   return_counts=True)
        ys_sup, counts = ys_sup[1:], counts[1:] / np.sum(counts[1:])
        support_with_class = [[] for _ in ys_sup]
        query_with_class = [[] for _ in ys_sup]

        for i, y in enumerate(ys_sup):
            for sent in support_sentences:
                sentence = copy.deepcopy(sent)
                for word in sentence:
                    if word[1][2:] != y:
                        word[1] = 'O'
                if np.unique([xy[1] for xy in sentence]).shape[0] > 1:
                    support_with_class[i].append(sentence)

            for sent in query_sentences:
                sentence = copy.deepcopy(sent)
                for word in sentence:
                    if word[1][2:] != y:
                        word[1] = 'O'
                query_with_class[i].append(sentence)

        # Here we are going to generate batches for training.
        super_list = []
        for episode in range(n_episodes):
            # To make it reproducible and flexible to debug
            np.random.seed(episode)
            query = []
            if np.random.rand() < valid_part:
                # With some probability we generate batch using target class
                while len(query) < (total_sentences - train_train_sentences):
                    np.random.shuffle(validation_batch)
                    # We repeat the number of support sentences to make the size of this set
                    # to be equal train_train_sentences. In fact this is a technical hack. We need it because it's hard
                    # to vary "support" size in the model during training
                    support = validation_batch[:(train_sentences // 2)] * (
                        train_train_sentences // (train_sentences // 2))
                    ys_here = np.concatenate([[xy[1] for xy in sentence] for sentence in support])
                    ys_sup = np.unique(ys_here)
                    # We add our empty sentences to the second 10 sentences to balance the set
                    query_candidate = validation_batch[(train_sentences // 2):] + empty_valid
                    np.random.shuffle(query_candidate)
                    query = []
                    # We do the cycle below to prevent the case when there are no 'I'-tags in the first 10 sentences
                    # and there are 'I'-tags in the second 10 sentences
                    for sentence in query_candidate:
                        ys_here = [xy[1] for xy in sentence]
                        if all([y in ys_sup for y in ys_here]):
                            query.append(sentence)
                            if len(query) == (total_sentences - train_train_sentences):
                                break
                # We add this batch to all the batches
                batch = support + query
                super_list += batch
            else:
                # With some probability we generate batch from the training set
                class_now = np.random.choice(counts.shape[0], p=counts)
                while len(query) < (total_sentences - train_train_sentences):
                    # We just sample the chosen number of support sentences
                    support = np.random.choice(support_with_class[class_now], size=train_train_sentences,
                                               replace=False).tolist()
                    ys_here = np.concatenate([[xy[1] for xy in sentence] for sentence in support])
                    ys_sup = np.unique(ys_here)
                    np.random.shuffle(query_with_class[class_now])
                    # We do the cycle below to prevent the case when there are no 'I'-tags in the support sentences
                    # and there are 'I'-tags in the query sentences
                    for sentence in query_with_class[class_now]:
                        ys_here = [xy[1] for xy in sentence]
                        if all([y in ys_sup for y in ys_here]):
                            query.append(sentence)
                        if len(query) == total_sentences - train_train_sentences:
                            break
                # We add this batch to all the batches
                super_list += support + query
    return super_list


@DatasetReader.register("onto_pnet")
class PnetOntoDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 valid_class: str = None,
                 random_seed: int = None,
                 drop_empty: bool = False,
                 valid_part: float = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.valid_class = valid_class
        self.random_seed = random_seed
        self.drop_empty = drop_empty
        self.valid_part = valid_part
        self.feature_labels = set(feature_labels)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # Here we just pass all the parameters to dataset reader
        if file_path[-8:] == 'test.txt':
            data = snips_reader('test.txt', valid_class=self.valid_class, random_seed=self.random_seed,
                                drop_empty=self.drop_empty, valid_part=self.valid_part)
        elif file_path[-9:] == 'train.txt':
            data = snips_reader('train.txt', valid_class=self.valid_class, random_seed=self.random_seed,
                                drop_empty=self.drop_empty, valid_part=self.valid_part)
        else:
            data = snips_reader('valid.txt', valid_class=self.valid_class, random_seed=self.random_seed,
                                drop_empty=self.drop_empty, valid_part=self.valid_part)

        for fields in data:
            # unzipping trick returns tuples, but our Fields need lists

            tokens, ner_tags = [list(field) for field in zip(*fields)]
            # TextField requires ``Token`` objects
            tokens = [Token(token) for token in tokens]
            sequence = TextField(tokens, self._token_indexers)

            instance_fields: Dict[str, Field] = {'tokens': sequence}
            # Add "feature labels" to instance
            if 'ner' in self.feature_labels:
                instance_fields['ner_tags'] = SequenceLabelField(ner_tags, sequence, "ner_tags")
            # Add "tag label" to instance
            instance_fields['tags'] = SequenceLabelField(ner_tags, sequence)
            yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'PnetOntoDatasetReader':
        # token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True),
                          "token_characters": TokenCharactersIndexer(),
                          "elmo": ELMoTokenCharactersIndexer()
                          }
        valid_class = params.pop('valid_class')
        random_seed = params.pop('random_seed')
        drop_empty = params.pop('drop_empty')
        valid_part = params.pop('valid_part')

        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return PnetOntoDatasetReader(token_indexers=token_indexers,
                                     valid_class=valid_class,
                                     random_seed=random_seed,
                                     drop_empty=drop_empty,
                                     valid_part=valid_part,
                                     tag_label=tag_label,
                                     feature_labels=feature_labels,
                                     lazy=lazy)
