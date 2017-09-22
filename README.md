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

A sample run command:

```bash
python3 app/train.py --use-attention --tag v3 --num-steps 360000 --batch-size 128 --steps-per-eval 30000 --learning-rate 0.0005
```

## Run

When you have a trained model, run `app/chatbot.py` can start a chatbot using you model.

Note: you have to change the path in `app/chatbot.py` to your model path.

Here is a sample output of the chatbot:

    > hi
    oh , honey ?
    > what are you doing ?
    watch my to to .
    > i love you
    i love you really . <pad> <pad> <pad> <pad> <pad> pony <pad> gay loser barney content <pad> <pad> <pad> durden bless imbecile caring
    > i feed bad
    how ? swann ? ? ? <pad> <pad> <pad> <pad> impressive -
    > i am happy for you
    you you you happy
    > yes , i am happy for you .
    why may to to understand
    > you look so stupid !
    i ruined a sorry .
    > are you sorry ?
    yes . . .
    > are you stupid ?
    no .
    > are you a computer ?
    please .
    > happy birthday
    sherry . wow <pad> <pad> ward necessarily boatwright
    > happy birthday !
    yes .
    > happy birthday ?
    yes .
    > happy birthday .
    hunh huh
    > sure .
    why ? dinner <pad>

## reference

This project is inspired by [Stanford CS20si assignment 3](http://web.stanford.edu/class/cs20si/).

The implementation refer to [Google Seq2Seq project](https://google.github.io/seq2seq/).

