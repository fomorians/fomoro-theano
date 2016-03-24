# Fomoro Theano + Lasagne Starter

Starter project for the [getting started](https://fomoro.gitbooks.io/guide/content/getting_started.html) guide. Based on [this Lasagne tutorial](http://lasagne.readthedocs.org/en/latest/user/tutorial.html).

## Training

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for Fomoro.
2. Clone the repo: `git clone https://github.com/fomorians/fomoro-theano.git && cd fomoro-theano`
3. Create a new model: `fomoro model create`
4. Start training: `fomoro session start`
5. Follow the logs: `fomoro session logs -f`

### Local Setup

1. [Install Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html).
2. Clone the repo: `git clone https://github.com/fomorians/fomoro-theano.git && cd fomoro-theano`
3. Run training: `python main.py`
