import os
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description='Run script for threestudio')
    parser.add_argument('--config_input', default='configs/texture_transfer/transfer_to_bunny.yaml')
    parser.add_argument('--config_output', default='configs/texture_transfer/meta_chair2_multi.yaml')
    parser.add_argument('--shape_path', default='shapes/bunny.obj')
    parser.add_argument('--diffusion_path', default='tuned_models/teapot_model/checkpoint')
    args = parser.parse_args()

    with open(args.config_input, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['log']['exp_name'] = '%s_as_%s' % (
        os.path.basename(args.config_output).replace('.yaml', ''), 
        os.path.basename(args.shape_path).replace('.obj', ''))
    config_dict['guide']['diffusion_name'] = args.diffusion_path
    config_dict['guide']['shape_path'] = args.shape_path

    with open(args.config_output, 'w') as file:
        yaml.dump(config_dict, file)

    print('New config file is saved at %s!' % args.config_output)

if __name__ == "__main__":
    main()