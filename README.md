# MAML Same-Different Learning

## Setup
1. Install requirements:

pip install torch learn2learn h5py tqdm

2. Generate meta-learning datasets:

python meta_data_generator_h5.py

3. Train MAML:

python maml_same_different.py


## Configuration
Key parameters in maml_same_different.py:
- meta_batch_size: Number of tasks per batch
- support/query sizes: we have 9 same/different training tasks, so these are 18 and 9 respectively 
- adapatation_steps: iterations over the support set 
- learning_rates: 0.01 (inner), 0.001 (outer) - need to do hyperparemeter tuning 
