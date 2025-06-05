# 8107
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/open_door 

# 8107
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/microwave

# 8105
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/drawer
# 8105
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/pick_and_place

# sichang
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/open_door ft_denoising_steps=10

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/drawer ft_denoising_steps=10

# sichang
python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/microwave ft_denoising_steps=10

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/pick_and_place ft_denoising_steps=10





