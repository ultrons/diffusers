export dataset_name="lambdalabs/pokemon-blip-captions"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"


#export PJRT_DEVICE=TPU
# Default is 0, set to 1 for debugging
export PT_XLA_DEBUG=0
export USE_TORCH=ON
export TPU_NUM_DEVICES=4

#These last two exports are not necessary, but make for more helpfully labeled profiles:

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

#python3 -u ~/transformers/examples/pytorch/xla_spawn.py    \
#--num_cores 4 train_text_to_image.py    \
python3 train_text_to_image_xla.py \
--pretrained_model_name_or_path=$MODEL_NAME    \
--dataset_name=$dataset_name    \
--use_ema    \
--resolution=512  \
--center_crop  \
--random_flip    \
--train_batch_size=1    \
--gradient_accumulation_steps=1    \
--gradient_checkpointing    \
--max_train_steps=15000    \
--learning_rate=1e-05    \
--max_grad_norm=1    \
--lr_scheduler="constant"  \
--lr_warmup_steps=0    \
--output_dir="sd-pokemon-model"
