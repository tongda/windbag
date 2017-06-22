# windbag
A chatbot based on Deep Learning. Mainly for proof of concept, not for production.

## Train

```bash
$ python app/train.py --help
usage: train.py [-h] [--use-attention] [--num-steps NUM_STEPS]
                [--write-summary] [--tag TAG]

Windbag trainer.

optional arguments:
  -h, --help            show this help message and exit
  --use-attention       Flag of whether to use attention.
  --num-steps NUM_STEPS
                        Number of steps.
  --write-summary       Flag of whether to write summaries.
  --tag TAG             Tag of experiment.
```

## reference

This project is inspired by [Stanford CS20si assignment 3](http://web.stanford.edu/class/cs20si/).

The implementation refer to [Google Seq2Seq project](https://google.github.io/seq2seq/).

