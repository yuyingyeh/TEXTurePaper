import os
import argparse
from datetime import datetime
import glob

def main():
    parser = argparse.ArgumentParser(description='Run script for threestudio')
    parser.add_argument('--texturepaper_root', default='/openroomssubstance/yyeh/TEXTurePaper')
    parser.add_argument('--dreambooth_name', default='meta_chair2_multi')
    # parser.add_argument('--shape_name', default='bunny')
    parser.add_argument('--shape_list', nargs='*', type=str, help='Input a list of shape names')
    parser.add_argument('--mask_type', default=None, help='None, white or random')
    parser.add_argument('--threestudio_root', default='/openroomssubstance/yyeh/threestudio')
    parser.add_argument('--u2net_root', default='/openroomssubstance/yyeh/U-2-Net')
    args = parser.parse_args()

    mesh_data_dir_list = [
        '/yyeh-data/threestudio/meshes/%s.obj',
        '/yyeh-data/3D-FUTURE/3D-FUTURE-model/%s/normalized_model.obj',
        # '/mnt/graphics_ssd/home/yyyeh/GitRepo/TEXTurePaper/shapes/%s.obj', 
        # '/mnt/graphics_ssd/home/yyyeh/GitRepo/threestudio/meshes/%s.obj',
        # '/mnt/graphics_ssd/home/yyyeh/data/scene_geometry/%s.obj', 
        # '/mnt/graphics_ssd/home/yyyeh/data/3D-FUTURE-model/%s/normalized_model.obj',
    ]
    mesh_id_dict = {'room_0': 'Mesh008', 
                    'bed_0': '02133f42-f8b1-4b10-9fe2-dec9a0bd1325',
                    'bed_1': '48d2dfa5-55dd-408f-a33b-4a085742e055',
                    'bed_2': '3f3e9fe3-3db0-407b-9956-57025c5b7e6d',
                    'bed_3': '647d9600-d011-409f-af3a-68afe22dd8cd',
                    'bed_4': '12b2dd21-c308-46e1-a375-471f67c2af77',
                    'bed_5': '0ab8bade-d500-4b65-ae65-020136a85d1e',
                    'sofa_0': 'f89da2db-ad8c-4582-b186-ed2a46f3cb15',
                    'sofa_1': '614ebd4c-9540-4ab0-85ff-0998020ce928',
                    'sofa_2': '3e08cbe7-87b9-45d6-80a4-0dfaee2b025c',
                    'sofa_3': 'ea720506-d8df-408c-a127-61b20b1a44b1',
                    'sofa_4': '517709cf-79a5-3309-af40-05e0c5ec992c',
                    'sofa_5': '1e71562b-34e3-44ac-b28e-d17589060ad0',
                    'sofa_6': '6132dd65-21cf-3e28-9ab3-4ea659d11be5',
                    'sofa_7': '592abfe3-905a-4d5b-98e7-f8d27568b531',
                    }
    
    ############## APPLY MASK #####################
    if args.mask_type is None or args.mask_type == 'None':
        dreambooth_name = args.dreambooth_name
    else:
        assert (args.mask_type in ['white', 'random'])
        dreambooth_name = '%s_%s' % (args.dreambooth_name, args.mask_type)
        cmd = 'python3 run_segmentation_bg_aug.py --dreambooth_name %s --mask_type %s' % (args.dreambooth_name, args.mask_type)
        cmd += ' --u2net_root %s --threestudio_root %s' % (args.u2net_root, args.threestudio_root)

        os.chdir(args.threestudio_root)
        print('>>> Running Command:')
        print(cmd)
        os.system(cmd)

    ############## APPLY MASK #####################
    # if args.mask_type is None or args.mask_type == 'None':
    #     dreambooth_name = args.texture_name
    # else:
    #     assert (args.mask_type in ['white', 'random'])
    #     dreambooth_name = '%s_%s' % (args.texture_name, args.mask_type)
    #     cmd = 'python3 run_segmentation_bg_aug.py --dreambooth_name %s --mask_type %s' % (args.texture_name, args.mask_type)
    #     os.chdir(args.threestudio_root)
    #     print('>>> Running Command:')
    #     print(cmd)
    #     os.system(cmd)

    cmd = ''

    # 1. Render Training Data
    rendered_list = glob.glob('%s/images/%s/*.pt' % (args.texturepaper_root, dreambooth_name))
    if len(rendered_list) > 0:
        print('--> Dataset for %s exists! Skip rendering training data!' % dreambooth_name)
    else:
        cmd += 'python3 -m scripts.generate_data_from_images '
        cmd += '--images_dir=%s/dreambooth_imgs/%s ' % (args.threestudio_root, dreambooth_name)
        cmd += '--output_dir=images/%s; ' % dreambooth_name

    # 2. Diffusion Fine-Tuning
    if os.path.exists('%s/tuned_models/%s/checkpoint/model_index.json' % (args.texturepaper_root, dreambooth_name)):
        print('--> %s fine-tuned model exists! Skip finetuning!' % dreambooth_name)
    else:
        cmd += 'python3 -m scripts.finetune_diffusion '
        cmd += '--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth '
        cmd += '--instance_data_dir=images/%s ' % dreambooth_name
        cmd += '--instance_prompt="a photo of a <object>" '
        cmd += '--lr_warmup_steps=0 --max_train_steps=10000 --scale_lr --output_dir tuned_models/%s ' % dreambooth_name
        cmd += '--eval_path=configs/texture_transfer/eval_data.json; '

        os.chdir(args.texturepaper_root)
        print('>>> Running Command:')
        print(cmd)
        os.system(cmd)

    # 3. Generate texture
    # Generate config file for next step
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config = 'configs/texture_transfer/%s-as-%s@%s.yaml' % (dreambooth_name, args.shape_name, timestamp)
    if args.shape_list:
        for shape_name in args.shape_list:
            # /mnt/graphics_ssd/home/yyyeh/GitRepo/TEXTurePaper/experiments/mug_rabbit0_multi_random-as-mug/results/step_00010_texture.png
            exp_name = '%s-as-%s' % (dreambooth_name, shape_name)
            # if os.path.exists(os.path.join(args.repo_root, 'experiments', exp_name, 'results/step_00010_texture.png')):
            if os.path.exists(os.path.join(args.texturepaper_root, 'experiments', exp_name, 'results/imgs/step_00010_0000_rgb.png')):
                print('Exp %s exists! Skip!' % exp_name)
                continue
            mesh_id = shape_name if shape_name not in mesh_id_dict.keys() else mesh_id_dict[shape_name]
            isExist = False
            for mesh_data_dir in mesh_data_dir_list:
                shape_path = mesh_data_dir % mesh_id
                if os.path.exists(shape_path):
                    isExist = True
                    break
            if not isExist:
                assert False

            config = 'configs/texture_transfer/%s.yaml' % (exp_name)
            cmd = 'python3 generate_config.py --config_input=configs/texture_transfer/template_full.yaml '
            cmd += '--config_output=%s ' % (config)
            cmd += '--shape_path=%s ' % shape_path
            cmd += '--diffusion_path=tuned_models/%s/checkpoint; ' % dreambooth_name
            
            # 3. Run TEXTure with Personalized Model
            cmd += 'python3 -m scripts.run_texture --config_path=%s ' % (config)
            
            # [for nerfacc] to make sure .cache folder can be created with permission
            # os.environ["HOME"] = args.home
            # os.environ["TORCH_HOME"] = args.torch_home
            
            os.chdir(args.texturepaper_root)
            print('>>> Running Command:')
            print(cmd)
            os.system(cmd)

if __name__ == "__main__":
    main()