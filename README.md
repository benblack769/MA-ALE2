### Install

```
pip install git+https://github.com/cpnota/autonomous-learning-library/tree/develop
```

### Run

```
python3 plot_all_one.py plot_data/builtin_results.csv --vs-builtin
```

Generates the plots vs the builtin agent

```
python3 plot_all_one.py plot_data/all_out.txt  
```

Generates the plots vs the random agent

Plots can be found near the data file, i.e. `plot_data/all_out.txt.png`


### Files

* Environment code
    * all_envs.py  
        * contains list of pettingzoo environments that should be trained
    * env_utils.py
        * contains environment preprocessing coe
* Policy code (each one of these has a function that makes a trainable ALL agent).
    * shared_ppo.py
    * shared_rainbow.py
    * shared_utils.py
    * shared_vqn.py (not used/working!)
    * independent_rainbow.py
    * independent_ppo.py
    * model.py
        * some experimental models to use for policies
*  Training code
    * experiment_train.py
        * trains agent returned by policy code
    * gen_train_runs.py
        * Generates many command line calls to experiment_train.py so that many experiments can be run with Slurm, Kabuki, or another job execution service.
* Evaluation code
    * experiment_eval.py
        * evaluates agent returned by checkpoint
        * can evaluate vs random agent, vs builtin agent (on specific environments), or vs trained opponent.
    * ale_rand_test.py
        * evaluates random agent vs random agent on all the environments, reports the results in a json file
    * ale_rand_test_builtin.py
        * evaluates random agent vs builtin agent on all the environments, reports the results in a json file
    * generate_evals.py
        * generates many calls to experiment_eval.py so that the evaluation jobs can be run with Slurm, Kabuki, or another job execution service
* Plotting code
    * plot_all_one.py
        * Looks at input csv file, specific random data file inside plot_data folder, and generates a plots with the results
* Neural Fictitious self-play code (none of it fully working)
    * nfsp_models.py
        * some experimental models to use for NFSP
    * neural_fictitious_selfplay.py
        * Preliminary code with some NFSP logic. Not clear if it is worth using.
