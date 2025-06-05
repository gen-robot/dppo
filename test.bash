python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/open_door 

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/drawer

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/microwave

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/pick_and_place

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/open_door ft_denoising_steps=10


python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/drawer ft_denoising_steps=10

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/microwave ft_denoising_steps=10

python script/run.py --config-name=ft_ppo_diffusion_img \
    --config-dir=cfg/sapien/finetune/pick_and_place ft_denoising_steps=10





