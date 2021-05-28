# DAPT
> This reposity contains codes for paper "**_Differentiable Prompt Makes Pre-trained LanguageModels Better Few-shot Learners_**"
## Requirements
- Python=3.6
- Install dependencies by `pip install -r requirements.txt`.
## Data Preparation
- Prepare processed 16-shot GLUE datasets in `data` folder by following the instruction [here](https://github.com/princeton-nlp/LM-BFF#prepare-the-data).
- This reposity also supports training on SuperGLUE datasets from paper ["**_GPT Understands, Too_**"](https://arxiv.org/abs/2103.10385), and you can put the data folder [`FewGLUE_32dev`](https://github.com/THUDM/P-tuning/tree/main/FewGLUE_32dev) in `data` folder.
## How to Run
### Quick Start
- To run with default setting (whose parameters might not be optimal), use `run.py`:
  - In this paper's setting, please specify `encoder=inner2` or `encoder=inner`, where `2` stands for two stage training in paper;
  - For manual prompt setting, `encoder=manual`;
  - For P-Tuning setting in paper ["**_GPT Understands, Too_**"](https://arxiv.org/abs/2103.10385), use `encoder=lstm`.

```bash
$ python run.py -h
usage: run.py [-h] [--encoder {manual,lstm,inner,inner2}] [--task TASK]
              [--num_splits NUM_SPLITS] [--repeat REPEAT] [--load_manual]
              [--extra_mask_rate EXTRA_MASK_RATE]
optional arguments:
  -h, --help            show this help message and exit
  --encoder {manual,lstm,inner,inner2}
  --task TASK
  --num_splits NUM_SPLITS
  --repeat REPEAT
  --load_manual
  --extra_mask_rate EXTRA_MASK_RATE
```
### Search for Optimal Hyper Parameters with W&B
- Note the results reported in our paper follows the same evaluation protocol in paper ["**_Making Pre-trained Language Models Better Few-shot Learners_**"](https://arxiv.org/pdf/2012.15723.pdf), by aggregating the best result of each seed split through parameter grid search.
- To reproduce the results in our paper, please register at [wandb](https://wandb.ai/) and use api_key to login and run `sweep.py`:

```bash
$ wandb login
# enter your api_key to login to wandb service...
$ python sweep.py -h
usage: sweep.py [-h]
                [--task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}]
                [--encoder {none,mlp,lstm,inner,inner2}]
                [--seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]]
                [--batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]]
                [--grad_accumulation_steps {1,2,4}]
                [--extra_mask_rate EXTRA_MASK_RATE] [--sweep_id SWEEP_ID]
optional arguments:
  -h, --help            show this help message and exit
  --task {SST-2,sst-5,mr,cr,mpqa,subj,trec,CoLA,MNLI,MNLI-mm,SNLI,QNLI,RTE-glue,MRPC,QQP}
  --encoder {none,mlp,lstm,inner,inner2}
  --seed_split {13,21,42,87,100} [{13,21,42,87,100} ...]
  --batch_size {4,8,16,24,32} [{4,8,16,24,32} ...]
  --grad_accumulation_steps {1,2,4}
  --extra_mask_rate EXTRA_MASK_RATE
  --sweep_id SWEEP_ID
```
- And you can view and gather optimal results in detail at [wandb](https://wandb.ai/).
### Specified Running
- To run with specified parameters, you can interact with `cli.py`.
- Put task `SST-2` as an example (will perform traininng on 1 split at a time):

```bash
export CUDA_VISIBLE_DEVICES=0 &&
python3 cli.py \
--data_dir data/k-shot/SST-2/16-13 \
--model_type albert \
--model_name_or_path roberta-large \
--cache_dir pretrain/roberta-large \
--task_name sst-2 \
--output_dir output/sst-2-inner2 \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 250 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "inner" \
--two_stage_train
```
- After training, you can view the results in `output/sst-2-inner2/result.txt`.
