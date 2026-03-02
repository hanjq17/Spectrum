# algo=nocache 
algo=spectrum
gpu_ids=0
# can also use multiple gpus in parallel
# gpu_ids=0,1,2,3
ngpu=$(echo $gpu_ids | awk -F',' '{print NF}')
total_prompt_num=2000000

# algo hyperparams
w=0.5
lam=0.1
window_size=2
flex_window=0.75

model_name=flux
# model_name=sd3-5

output_base_path="cache_exp_outputs"
prompt_file=prompts/DrawBench200.txt
exp_name=${model_name}/${algo}_window${window_size}_w${w}_lam${lam}_flex${flex_window}
echo "Running experiment: ${exp_name}"

CUDA_VISIBLE_DEVICES=${gpu_ids} \
python src/text_to_image.py \
    algo=${algo} \
    algo.w=${w} \
    algo.lam=${lam} \
    window_size=${window_size} \
    flex_window=${flex_window} \
    exp_name=${exp_name} \
    ngpu=${ngpu} \
    total_prompt_num=${total_prompt_num} \
    model=${model_name} \
    output_base_path=${output_base_path} \
    prompt_file=${prompt_file}
