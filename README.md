# PentominoExps

This is a repo for the experiments in,
     Caglar Gulcehre, Yoshua Bengio, "Knowledge Matters: Importance of Prior Information for Optimization", JMLR, 2015.

By using the codes in this repo, it is possible to reproduce the experiments in.

To be able to reproduce our results, you need to have the specific versions of the following python packages:
    * Numpy: 1.9.2
    * Theano: 0.7.0.dev-079181cf9e503d61cb9cd830ddc87c81b01fbc6b
    * Scipy: 0.16.0

In order to run the experiments for the training with hints, go under the Experiments folder and execute:

./run.sh run_model_hints.py --pento_dir <directory_to_the_datasets>

To run the experiments for the training without hints, go under the Experiments folder and execute:

./run.sh run_model_nohints.py --pento_dir <directory_to_the_datasets>

To plot the learning curves in the paper, use the plot_learning_curves.py script under experiments/
directory.

The datasets are available at:

http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/pentomino/datasets/
