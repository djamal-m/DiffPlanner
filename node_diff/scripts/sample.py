import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import torch as th
import json

from node_diff import dist_util, logger
from node_diff.rplan_datasets import load_rplan_data
from node_diff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)


dataset_json_dir = '../../dataset/dataset_json'
with open(f'{dataset_json_dir}/data_test.json', encoding='utf-8') as f:
    dataset_json_ini = json.load(f)

name2index_dict = {}
for entry_i in range(len(dataset_json_ini)):
    name_fp = dataset_json_ini[entry_i].get("name")
    name2index_dict[name_fp] = entry_i

outputs = []


def get_datajson_from_tensor(data_tensor, cond, is_syn, support_boundary, support_conditions, support_partial):

    data_json = {}
    prefix = 'syn_' if is_syn else ''

    data_json["name"] = cond['name_fp']
    data_json_ini = dataset_json_ini[name2index_dict[data_json["name"]]]

    if support_boundary:
        data_json["boundary"] = data_json_ini["boundary"]
        data_json["boundary_expand"] = data_json_ini["boundary_expand"]
        data_json["entrance_expand"] = data_json_ini["entrance_expand"]

    rooms_category = cond[f'{prefix}cond_category']
    num_rooms_gt = np.sum(np.any(rooms_category!=0, axis=1))

    rooms_category_list = np.argmax(rooms_category, axis=1).tolist()

    if support_partial:
        cond_partial = cond[f'{prefix}cond_partial']
        indices_pred = np.where(np.all(cond_partial==-1, axis=1))[0]
        cond_partial[indices_pred, :] = data_tensor[indices_pred, :]
        data_tensor = cond_partial

    canvas_size = 256
    canvas_area = canvas_size * canvas_size

    rooms_info = []
    room_id = 0
    for i in range(data_tensor.shape[0]):

        if support_conditions == '':
            is_room = 0 if data_tensor[i][0] < 0.5 else 1
            category = round(((data_tensor[i][1] + 1) / 2) * 6) - 1
        if support_conditions == 'n':
            is_room = 0 if i >= num_rooms_gt else 1
            category = round(((data_tensor[i][0] + 1) / 2) * 6) - 1
        if support_conditions == 'nc':
            is_room = 0 if i >= num_rooms_gt else 1
            category = int(rooms_category_list[i])

        if is_room != 1:
            continue
        
        room_info = {}
        room_info["id"] = room_id
        room_info["category"] = category
        room_info["size"] = round(((data_tensor[i, -3] + 1) / 2) * canvas_area)
        room_info["location"] = np.round(((data_tensor[i, -2:] + 1) / 2) * canvas_size).astype(int).tolist()
        
        if support_partial:
            partial_input_room = 1
            if i in indices_pred:
                partial_input_room = 0
            room_info["partial_input_node"] = partial_input_room

        rooms_info.append(room_info)

        room_id += 1
        
    data_json["rooms"] = rooms_info

    return data_json


def save_samples(
        sample, model_kwargs, 
        support_boundary, support_conditions, support_partial,
        save_gif=False, is_syn=False):

    if not save_gif:
        sample = sample[-1:]

    for i in range(sample.shape[1]):
        for k in range(sample.shape[0]):
            
            cond_i = {
                key: value[i].clone().cpu().numpy() if isinstance(value[i], th.Tensor) else value[i]
                for key, value in model_kwargs.items()
            }

            try:
                output = get_datajson_from_tensor(sample[k][i].clone().cpu().numpy(), cond_i, is_syn, support_boundary, support_conditions, support_partial)
                outputs.append(output)
            except Exception as e:
                print(f"error name_fp: {cond_i['name_fp']}")

    return 0


def main():

    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f'args.model_path: {args.model_path}')
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    for _ in range(1):
        logger.log("sampling...")
        tmp_count = 0

        if args.dataset == 'rplan':
            data = load_rplan_data(
                batch_size=args.batch_size,
                set_name=args.set_name,
                support_boundary=args.support_boundary,
                support_conditions=args.support_conditions,
                support_partial=args.support_partial
            )
        else:
            print("dataset does not exist!")
            assert False

        print(f'args.num_samples: {args.num_samples}')
        while tmp_count < args.num_samples:

            model_kwargs = {}

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            data_sample, model_kwargs = next(data)

            for key, value in model_kwargs.items():
                if isinstance(value, list):
                    model_kwargs[key] = [item.cuda() if isinstance(item, th.Tensor) else item for item in value]
                elif isinstance(value, th.Tensor):
                    model_kwargs[key] = value.cuda()

            sample = sample_fn(
                model,
                data_sample.shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            
            #sample_gt = data_sample.cuda().unsqueeze(0)
            #sample_gt = sample_gt.permute([0, 1, 3, 2])
            sample = sample.permute([0, 1, 3, 2])
            
            save_samples(sample, model_kwargs, args.support_boundary, args.support_conditions, args.support_partial, is_syn=True)
            
            tmp_count += sample.shape[1]

        logger.log("sampling complete")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    conditions_for_filename = ''
    if args.support_boundary:
        conditions_for_filename += 'b'
    conditions_for_filename += args.support_conditions
    if args.support_partial:
        conditions_for_filename += 'p'
        conditions_for_filename += '_25_node'

    if conditions_for_filename == '':
        conditions_for_filename = '_'

    output_json_path = f'{output_dir}/{conditions_for_filename}.json'

    openw = open(output_json_path, 'w')
    json.dump(outputs, openw, ensure_ascii=False)
    openw.close()

    return 0


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=1024,
        use_ddim=False,
        model_path='',
        output_dir='../../output/output_json',
        support_boundary=True,
        support_conditions='', # '', 'n', 'nc'
        support_partial=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()


