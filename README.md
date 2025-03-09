# MAML Same-Different Learning

## Setup
1. Install requirements:
pip install -r requirements.txt

2. python meta_data_generator_h5.py
Generates meta-learning datasets (NOTE: the same/different tasks takes up quite a lot of disk space, so generating this on a server is usually preferred)


3.
python run_pb_baselines.py
Trains baseline CNN's from Kim et al. study (2, 4, and 6 layer CNN's) (vanilla SGD, training on all same/different tasks from the [Puebla/Bowers study]([url](https://jov.arvojournals.org/article.aspx?articleid=2783637&__cf_chl_tk=eTasMRhqVWyQpG3O5PDBEaVH..nACRhKO5GaOyL_Mz4-1741031232-1.0.1.1-wiM5lsuw3UA6JT2vhePd1Zl9.5TccA73nSt6lQ4vXVg)) and testing on new examples of each task (in-distribution) 


4.python train_and_test_meta_baselines.py
Trains meta_baseline - same architectures as above, trained with MAML, tested on new episodes of each task (in-distribution) 

6. python experiment1_leave_one_out.py
Hold-out meta-learning - training just the highest performing CNN from baselines (6-layer CNN) on all P/B tasks except one held-out task (out-of-distribution)



## Configuration
Key parameters in maml_same_different.py:

meta_batch_size: Number of tasks per batch
support/query sizes: we have 9 same/different training tasks, and we vary support set sizes over [4,6,8,10], w/ fixed query size = 3.
adapatation_steps: iterations over the support set
learning_rates: 0.05 (inner), 0.001 (outer) 
