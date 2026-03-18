## Installation
```bash
conda create -n lyaplock python == 3.9.7
pip install -r requirements.txt
```

## Data Preparation
The relevant datasets need to be downloaded from [https://rome.baulab.info/data](https://rome.baulab.info/data) to the local `./data` folder.

## Edit
### 1. Edit LLAMA3-8B Model on CounterFact using LyapLock
```bash
python3 -m experiments.evaluate \
    --alg_name=LyapLock \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct` \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=10000 \
    --num_edits=100 \
    --downstream_eval_steps=10 \
    --alpha 60
```

Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ LyapLock/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_9999.json
```

#### 2. Summarize the results
```bash
python summarize.py --dir_name=AlphaEdit --runs=run_<run1>,run_<run2>
```


## Acknowledgment
Our code is based on  [``AlphaEdit``](https://github.com/jianghoucheng/AlphaEdit.git), [``MEMIT``](https://github.com/kmeng01/memit.git) and [``EMMET``](https://github.com/scalable-model-editing/unified-model-editing.git).