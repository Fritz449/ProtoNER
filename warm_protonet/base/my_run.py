#!/usr/bin/env python
import logging
import os
import sys

from pnet_iterator import PnetIterator
from pnet_model import PnetTagger
from pnet_ontonotes import PnetOntoDatasetReader

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    main(prog="python -m allennlp.run")
