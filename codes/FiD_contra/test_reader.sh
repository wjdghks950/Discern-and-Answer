python test_reader.py \
        --model_path checkpoint/nq_base_640k_semi_parametric_disc_p75/checkpoint/step-640001 \
        --eval_data ../../DATA/corpus/dataset_pert_prompt_dev_256_new_fix.json \
        --per_gpu_batch_size 1 \
        --n_context 5 \
        --name nq_base_640k_semi_parametric_disc_p75 \
        --checkpoint_dir checkpoint \
        --perturb 0.35 \
        --model_setting semi_parametric_pert #[parametric, semi_parametric, semi_parametric_pert]
