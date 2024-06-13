# Discern-and-Answer

**Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise** [[Paper](https://arxiv.org/abs/2305.01579)] <br>
[Giwon Hong*](https://honggiwon.github.io/), [Jeonghwan Kim*](https://wjdghks950.github.io/), [Junmo Kang*](https://jm-kang.github.io/), [Sung-Hyon Myaeng](https://scholar.google.com/citations?user=6pdKebMAAAAJ&hl=ko), [Joyce Jiyoung Whang](https://bdi-lab.kaist.ac.kr/#)

## Contents
- [Install](#install)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)
- [GPT3.5](#gpt-3.5)

## Install

1. Clone the current repository
```bash
git clone https://github.com/wjdghks950/Discern-and-Answer.git
cd Discern-and-Answer
```

2. Install the conda enviroment
```bash
conda env create -f environment.yaml
```

3. Activate the conda enviroment
```bash
conda activate DaC_env
```

## Dataset

1. Download datasets from Download the dataset from the following link: [MacNoise dataset](https://drive.google.com/drive/folders/1icxbi5_9OvNkr4cG5h7NrY4F01Y7amj0?usp=sharing)

2. Place them under DATA folder

- e.g., Discern-and-Answer/DATA/corpus/NQ_train_titlemerged_longpre-corpus.json

3. Training dataset list

- `DATA/corpus/NQ_train_titlemerged_longpre-corpus.json` : The NQ (Natural Questions) train set used for finetuning the FiD model. This dataset is perturbed following the method by Longpre et al. (2021) and includes the top 100 documents per question.
- `DATA/corpus/NQ_train_titlemerged_chatgpt_top20.json` : The NQ (Natural Questions) train set used for finetuning the FiD model. This dataset is perturbed by our proposed method (MacNoise) and includes the top 20 documents per question.
- `DATA/corpus/NQ_train_titlemerged_joint_top20.json` : The NQ (Natural Questions) train set used for finetuning the FiD model. This dataset is perturbed following the method by Longpre et al. (2021) and our proposed method (MacNoise), and includes the top 100 documents per question (where top 20 documents have both Longpre & MacNoise perturbation).

4. Evaluation dataset list

- `DATA/corpus/NQ_eval_longpre_dev_256_new_fix.json` : The sampled NQ dev set with 256 instances used to evaluate the finetuned FiD model and GPT 3.5. This dataset is perturbed following the method by Longpre et al. (2021) and includes the top 5 documents per question.
- `DATA/corpus/NQ_eval_longpre_test_fix.json` : The NQ test set used to evaluate the finetuned FiD model and GPT 3.5. This dataset is perturbed following the method by Longpre et al. (2021) and includes the top 5 documents per question.
- `DATA/corpus/NQ_eval_gpt4_dev_256_new_fix.json` : The sampled NQ dev set with 256 instances used to evaluate the finetuned FiD model and GPT 3.5. This dataset is perturbed by our proposed method (MacNoise) and includes the top 5 documents per question.
- `DATA/corpus/TQA_eval_gpt4_dev_256_new_fix.json` : The sampled TQA dev set with 256 instances used to evaluate the finetuned FiD model and GPT 3.5. This dataset is perturbed by our proposed method (MacNoise) and includes the top 5 documents per question.

## Train

Configure training settings in `Discern-and-Answer/codes/FiD_contra/train_reader.sh`

~~~
python train_reader.py \
        --train_data ../../DATA/corpus/NQ_train_titlemerged_longpre-corpus.json \
        --eval_data ../../DATA/corpus/NQ_dev_titlemerged_longpre-corpus.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --total_steps 640000 \
        --accumulation_steps 64 \
        --eval_freq 800000 \
        --save_freq 160000 \
        --name nq_base_640k_semi_parametric_disc_p75 \
        --checkpoint_dir checkpoint \
        --perturb 0.75 \
        --model_setting semi_parametric_pert #[parametric, semi_parametric, semi_parametric_pert]
~~~

- `train_data / eval_data`: paths to the train/eval datasets. The list of possible datasets can be found above.
- `name`: model name to be saved in FiD_contra/checkpoint
- `perturb`: perturbation probability. Documents with perturbable answers are perturbed according to this probability. Note: The `--perturb` parameter in training refers to the probability of perturbing documents with perturbable answers. In evaluation, `--perturb` refers to the proportion of documents being perturbed out of the total. Therefore, setting `--perturb` to 0, 31, 50, 75 in training corresponds to 0, 15, 25, 35 in evaluation. (For more details, please refer to the paper.)
- `model_setting`: it should be one of the followings: "parametric", "semi_parametric", "semi_parametric_pert"

>`parametric`: Inferring an answer using only parametric knowledge of a model without using retrieved passages at all. Therefore, the performance is not affected by the perturbation probability.

>`semi_parametric`: The same settings as the original FiD, inferring answers using retrieved passages.

>`semi_parametric_pert`: Jointly train a discriminator that determines whether passages are perturbed or not in order to mitigate the effects of perturbed documents (Discriminator_FiD)

## Evaluation

Configure evaluation settings in `Discern-and-Answer/codes/FiD_contra/test_reader.sh`

~~~
python test_reader.py \
        --model_path checkpoint/nq_base_640k_semi_parametric_disc_p75/checkpoint/step-640001 \
        --eval_data ../DATA/corpus/NQ_eval_longpre_dev_256_new_fix.json \
        --per_gpu_batch_size 1 \
        --n_context 5 \
        --name nq_base_640k_semi_parametric_disc_p75 \
        --checkpoint_dir checkpoint \
        --perturb 0.35 \
        --model_setting semi_parametric_pert #[parametric, semi_parametric, semi_parametric_pert]
 ~~~

- `model_path`: path to model checkpoint
- `eval_data`: paths to the evaluation datasets with deterministic perturbation in DATA/Evaluation. The list of possible datasets can be found above.
- `perturb`: perturbation probability. It should be one of 0.0, 0.15, 0.25, 0.35. This is to use a pre-made perturbation for deterministic evaluation.
- `model_setting`: same as in training

## GPT-3.5
In order to handle frequent failures of the GPT-3.5 API, the code is provided in the form of a jupyter notebook.

1. Answer Generation

To generate answers through in-context learning of GPT-3.5, use scripts from `Discern-and-Answer/codes/GPT`

- `GPT_in-context_learning_nq_longpre.ipynb` : A script for generating answers using the GPT-3.5 model on the NQ dev/test set perturbed following the method by Longpre et al. (2021). During this process, the classification results of the dev/test set by the FiD model, trained on the corresponding train set, can be used.
- `GPT_in-context_learning_nq_gpt4.ipynb` : A script for generating answers using the GPT-3.5 model on the NQ dev set perturbed by our proposed method (MacNoise dataset). During this process, the classification results of the dev/test set by the FiD model, trained on the corresponding train set, can be used.
- `GPT_in-context_learning_tqa_gpt4.ipynb` : A script for generating answers using the GPT-3.5 model on the TQA dev set perturbed by our proposed method (MacNoise dataset). During this process, the classification results of the dev/test set by the FiD model, trained on the perturbed NQ train set, can be used.

You can do various configurations in the third cell.

- GPT3_api_key: API key to use GPT-3.5. Refer to https://beta.openai.com/
- is_dev: if True, it will generate answers on the dev dataset. Else, on the test dataset
- use_parametric_only: If True, generate answers using only the parametric knowledge of GPT-3 without providing the retrieved passages
- use_pert_aware_instruction: If True, add an perturbation-aware instruction to GPT-3.5's prompt to identify perturbed passages and ignore them in answer generation. (Discriminator_inst)
- use_discriminator_fid: If True, inject the perturbation classification results of FiD discriminator (Discriminator_FiD) into prompts instead of GPT-3's perturbed passage identification
- pert_ratio: perturbation probability. It should be one of 0.0, 0.15, 0.25, 0.35. This is to use a pre-made perturbation for deterministic evaluation.
- train_sample_idx: Decide which of the 5 train samples to use. An ensemble can be performed in Evaluation only when answers are derived for all 5 train samples.
- train_sample_path: path to train sample json file in DATA

2. Evaluation

To evaluate answers generated by GPT-3.5, use scripts from `Discern-and-Answer/codes/GPT`.

- `GPT_evaluation_nq_longpre.ipynb` : A script to evaluate answers generated by `GPT_in-context_learning_nq_longpre.ipynb`
- `GPT_evaluation_nq_gpt4.ipynb` : A script to evaluate answers generated by `GPT_in-context_learning_nq_gpt4.ipynb`
- `GPT_evaluation_tqa_gpt4.ipynb` : A script to evaluate answers generated by `GPT_in-context_learning_tqa_gpt4.ipynb`

You can do various configurations in the third cell. They are identical to the settings in Answer Generation scripts.

If you run all cells, you can get the best/average/worst results and ensemble results for 5 samples for each specified setting.