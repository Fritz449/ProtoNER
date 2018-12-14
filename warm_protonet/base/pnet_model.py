from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure
import numpy as np
from torch.autograd import Variable
from sklearn.semi_supervised import LabelSpreading

"""
This file is a changed analogous file "crf_tagger.py" from allennlp. So this code may look strange. But it works.
"""

@Model.register("pnet_tagger")
class PnetTagger(Model):
    """
    The ``PnetTagger`` is the tagger that is describled in the paper "Few-shot classification in Named Entity Recognition task"

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    dropout:  ``float``, optional (detault=``None``)
    constraint_type : ``str``, optional (default=``None``)
        If provided, the CRF will be constrained at decoding time
        to produce valid labels based on the specified type (e.g. "BIO", or "BIOUL").
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 constraint_type: str = None,
                 include_start_end_transitions: bool = True,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 cuda_device: int = -1) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder

        # This is our trainable parameter that is used as logit of 'O'-tag
        self.bias_outside = torch.nn.Parameter(torch.zeros(1) - 4., requires_grad=True)
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        # We also train scales in the embedding space for every class assuming that they may be different.
        self.scale_classes = torch.nn.Parameter(torch.ones(self.num_tags), requires_grad=True)

        self.encoder = encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.last_layer = TimeDistributed(Linear(self.encoder.get_output_dim(), 64))

        if constraint_type is not None:
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(constraint_type, labels)
        else:
            constraints = None

        self.crf = ConditionalRandomField(
                        self.num_tags, constraints,
                        include_start_end_transitions=include_start_end_transitions
                )
        self.loss = torch.nn.CrossEntropyLoss()

        self.cuda_device = cuda_device
        if self.cuda_device >= 0:
            self.text_field_embedder = self.text_field_embedder.cuda(self.cuda_device)
            self.encoder = self.encoder.cuda(self.cuda_device)
            self.last_layer = self.last_layer.cuda(self.cuda_device)
            self.elmo_weight = torch.nn.Parameter(torch.ones(1).cuda(self.cuda_device), requires_grad=True)
        self.span_metric = SpanBasedF1Measure(vocab,
                                              tag_namespace=label_namespace,
                                              label_encoding=constraint_type or "BIO")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

        self.hash = 0
        self.number_epoch = 0

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[str]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        if self.cuda_device >= 0:
            # Here we create some permanent GPU Variables because it's inefficient to create new GPU Variable every time
            if self.number_epoch == 0:
                self.tokens = tokens['tokens'].clone().cuda(self.cuda_device)
                self.token_characters = tokens['token_characters'].clone().cuda(self.cuda_device)
                self.mask = util.get_text_field_mask(tokens).clone().cuda(self.cuda_device)
                self.elmo_tokens = tokens['elmo'].clone().cuda(self.cuda_device)
            else:
                self.tokens.data = tokens['tokens'].data.cuda(self.cuda_device)
                self.token_characters.data = tokens['token_characters'].data.cuda(self.cuda_device)
                self.mask.data = util.get_text_field_mask(tokens).data.cuda(self.cuda_device)
                self.elmo_tokens.data = tokens['elmo'].data.cuda(self.cuda_device)
        else:
            self.tokens = tokens['tokens'].clone()
            self.token_characters = tokens['token_characters'].clone()
            self.mask = util.get_text_field_mask(tokens).clone()
            self.elmo_tokens = tokens['elmo'].clone()

        # To prevent memory overflow we compute embeddings using internal minibatches
        number = 25

        elmo_parts_input = [self.elmo_tokens[(i * number):min(((i + 1) * number), tokens['elmo'].data.shape[0])] for i
                            in range(int(np.ceil(tokens['elmo'].data.shape[0] / number)))]
        tokens_parts_input = [self.tokens[(i * number):min(((i + 1) * number), tokens['elmo'].data.shape[0])] for i in
                              range(int(np.ceil(tokens['tokens'].data.shape[0] / number)))]
        chars_parts_input = [self.token_characters[(i * number):min(((i + 1) * number), tokens['elmo'].data.shape[0])]
                             for i in range(int(np.ceil(tokens['token_characters'].data.shape[0] / number)))]

        results = [self.text_field_embedder(
            {'elmo': elmo_part, 'tokens': tokens_part, 'token_characters': chars_part})
                   for elmo_part, tokens_part, chars_part in
                   zip(elmo_parts_input, tokens_parts_input, chars_parts_input)]

        # Clean memory
        del elmo_parts_input[:]
        del tokens_parts_input[:]
        del chars_parts_input[:]
        embedded_text_input = torch.cat(results, dim=0)
        del results[:]

        mask = util.get_text_field_mask(tokens)

        # Here we apply dropout to embeddings
        if self.dropout:
            dropped = self.dropout(embedded_text_input)

        # We again split our data to compute new hidden layer
        dropped_parts = [dropped[(i * number):min(((i + 1) * number), tokens['elmo'].data.shape[0])] for i in
                         range(int(np.ceil(tokens['elmo'].data.shape[0] / number)))]
        del dropped
        mask_parts = [self.mask[(i * number):min(((i + 1) * number), tokens['elmo'].data.shape[0])] for i in
                      range(int(np.ceil(tokens['elmo'].data.shape[0] / number)))]
        results = [self.encoder(dropped_part, mask_part)
                   for dropped_part, mask_part in zip(dropped_parts, mask_parts)]
        del dropped_parts[:]
        del mask_parts[:]
        encoded_text = torch.cat(results, dim=0)

        # Again we apply dropout
        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        # Apply the last layer
        embeddings = self.last_layer(encoded_text)

        # Here we split our batch into support and query sentences.
        # This division depends on what we do now: training or testing.
        # This happens because we generate train and test datasets using separate procedures.
        if embeddings.requires_grad:
            split_i = 40
        else:
            split_i = 20

        # Here we split all the data
        tags_support = tags[:split_i]
        tags_query = tags[split_i:]
        uniq_support = np.unique(tags_support.cpu().data.numpy())
        support = embeddings[:split_i]
        query = embeddings[split_i:]
        support_mask = mask[:split_i]
        query_mask = mask[split_i:]

        # We will need numpy-masks
        mask_query = query_mask.data.cpu().numpy()
        mask_support = support_mask.data.cpu().numpy()

        # We want to map from tag numbers given by general dictionary to numbers inside this batch and vice versa.
        decoder = dict(zip(uniq_support, np.arange(uniq_support.shape[0])))
        encoder = dict(zip(np.arange(uniq_support.shape[0]), uniq_support))

        # Here we spread out our embeddings using tab labels
        embeds_per_class = [[] for _ in np.arange(np.unique(uniq_support).shape[0])]
        tags_numpy = tags_support.data.cpu().numpy()
        for i_sen, sentence in enumerate(support):
            for i_word, word in enumerate(sentence):
                if mask_support[i_sen, i_word] == 1:
                    tag = tags_numpy[i_sen][i_word]
                    if tag > 0:
                        embeds_per_class[decoder[tag]].append(word)

        # Here we compute embeddings
        prototypes = [torch.zeros_like(embeds_per_class[1][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / len(embeds_per_class[i])

        # We are going to compute logits for every class in data because we use constant-size CRF layer.
        # Logits are equal -100 by default because we want our objects to have 0-probabilities
        # for classes that are not used in this batch
        logits = Variable(torch.zeros((tags_query.shape[0], tags_query.shape[1], self.num_tags))) - 100.
        for i_sen, sentence in enumerate(query):
            for i_word, word in enumerate(sentence):
                if mask_query[i_sen, i_word] == 1:
                    logits[i_sen, i_word, 0] = self.bias_outside
                    for i_class in range(len(embeds_per_class))[1:]:
                        distance = torch.sum(torch.pow(word - prototypes[i_class], 2))
                        logits[i_sen, i_word, encoder[i_class]] = -distance * self.scale_classes[encoder[i_class]]

        # Compute prediction
        best_paths = self.crf.viterbi_tags(logits, query_mask)

        # Just get the tags and ignore the score.
        query_tags = [x for x, y in best_paths]

        output = {"mask": mask}

        # Use negative log-likelihood of the true tag sequence as loss.
        # we do the same things as we do in a basic CRF-tagger.
        log_likelihood = self.crf(logits.cuda(self.cuda_device), tags_query, query_mask)
        if embeddings.requires_grad:
            log_likelihood = log_likelihood.cuda(self.cuda_device)
        else:
            log_likelihood = log_likelihood.detach().cuda(self.cuda_device)
        output["loss"] = -log_likelihood

        # Compute one-hot answers to compute F1-metric of prediction.
        class_probabilities = logits * 0.
        for i, instance_tags in enumerate(query_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1
        self.span_metric(class_probabilities, tags_query, query_mask)

        self.number_epoch += 1
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace="labels")
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
            ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'PnetTagger':
        cuda_device = params.pop("cuda_device")
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        label_namespace = params.pop("label_namespace", "labels")
        constraint_type = params.pop("constraint_type", None)
        dropout = params.pop("dropout", None)
        include_start_end_transitions = params.pop("include_start_end_transitions", True)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   label_namespace=label_namespace,
                   constraint_type=constraint_type,
                   dropout=dropout,
                   include_start_end_transitions=include_start_end_transitions,
                   initializer=initializer,
                   regularizer=regularizer,
                   cuda_device=cuda_device)

    @overrides
    def load_state_dict(self, state_dict, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """
        # Here we reset some parameters that we don't need to load from warming
        state_dict.pop('text_field_embedder.token_embedder_tokens.weight', None)
        state_dict.pop('text_field_embedder.token_embedder_token_characters._embedding._module.weight', None)
        state_dict.pop('tag_projection_layer._module.weight', None)
        state_dict.pop('tag_projection_layer._module.bias', None)
        state_dict.pop('crf.transitions', None)
        state_dict.pop('crf._constraint_mask', None)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))



