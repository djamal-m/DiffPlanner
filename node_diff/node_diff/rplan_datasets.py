import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import copy


def load_rplan_data(
    batch_size,
    set_name,
    support_boundary,
    support_conditions,
    support_partial
):
    print(f"loading {set_name}")
    deterministic = False if set_name =='train' else True
    dataset = RPlanDataset(set_name, support_boundary, support_conditions, support_partial)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader


get_one_hot = lambda x, z: np.eye(z)[x]


def get_datatensor(set_name, data_json, max_num_rooms, canvas_size, support_boundary, support_conditions, support_partial):

    data_json = copy.deepcopy(data_json)

    if support_boundary:
        # condition boundary
        boundary_rplan = data_json["boundary_expand"]
        cond_boundary = (np.array(boundary_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
        cond_boundary = np.repeat(cond_boundary, max_num_rooms, axis=0)
        # condition door
        door_rplan = data_json["entrance_expand"]
        cond_door = (np.array(door_rplan).flatten().reshape(1, -1) / canvas_size * 2 ) - 1
        cond_door = np.repeat(cond_door, max_num_rooms, axis=0)
    else:
        cond_boundary = np.full((max_num_rooms, 80), -1.0)
        cond_door = np.full((max_num_rooms, 8), -1.0)

    # node
    rooms_info = data_json["rooms"]

    num_rooms = len(rooms_info)
    
    np.random.seed(None)
    np.random.shuffle(rooms_info)

    canvas_area = canvas_size * canvas_size

    ncsxy_rooms = [[1, (int(room["category"]) + 1) / 3 - 1, (room["size"] / canvas_area * 2) - 1, (room["location"][0] / canvas_size * 2) - 1, (room["location"][1] / canvas_size * 2) - 1] for room in rooms_info]
    ncsxy_rooms = np.array(ncsxy_rooms)
    array_to_add = np.full((max_num_rooms - ncsxy_rooms.shape[0], ncsxy_rooms.shape[1]), -1.0)
    ncsxy_rooms = np.concatenate((ncsxy_rooms, array_to_add), axis = 0)

    # condition number
    cond_number = np.zeros((max_num_rooms, max_num_rooms), dtype = np.int8)
    for r_i in range(num_rooms): 
        cond_number[r_i] = np.array([get_one_hot(r_i, max_num_rooms)])

    # condition_category
    num_category = 6
    cond_category = np.zeros((max_num_rooms, num_category), dtype=np.int8)
    for r_i in range(num_rooms):
        cond_category[r_i] = np.array([get_one_hot(int(rooms_info[r_i]["category"]), num_category)])

    target_tensor = ncsxy_rooms

    atten_mask = np.zeros((max_num_rooms, max_num_rooms))

    padding_mask = np.repeat(1, max_num_rooms)
    
    if support_conditions == 'n':
        target_tensor = ncsxy_rooms[:, 1:]
        atten_mask = np.ones((max_num_rooms, max_num_rooms))
        atten_mask[:num_rooms, :num_rooms] = 0

        padding_mask[num_rooms:max_num_rooms] = 0

    if support_conditions == 'nc':
        target_tensor = ncsxy_rooms[:, 2:]
        atten_mask = np.ones((max_num_rooms, max_num_rooms))
        atten_mask[:num_rooms, :num_rooms] = 0

        padding_mask[num_rooms:max_num_rooms] = 0

    cond_partial = np.full((target_tensor.shape), -1.0)
    if support_partial:
        random_k = random.randint(0, max_num_rooms-1)
        if set_name == 'test':
            ratio_partial = 0.25
            #ratio_partial = 0.5
            #ratio_partial = 0.75
            random_k = round(max_num_rooms * ratio_partial)

        if random_k > 0:
            cond_partial[:random_k, :] = target_tensor[:random_k, :]

    return target_tensor, atten_mask, padding_mask, cond_boundary, cond_door, cond_number, cond_category, cond_partial


class RPlanDataset(Dataset):

    def __init__(self, set_name, support_boundary, support_conditions, support_partial):

        super().__init__()

        self.set_name = set_name
        self.max_num_rooms = 8
        self.canvas_size = 256
        
        self.support_boundary = support_boundary # true or false
        self.support_conditions = support_conditions # '', 'n', 'nc'
        self.support_partial = support_partial # true or false

        dataset_json_path = f'../../dataset/dataset_json/data_{self.set_name}.json'
        with open(dataset_json_path, encoding='utf-8') as f:
            self.dataset_json = json.load(f)
        
    def __len__(self):
        return len(self.dataset_json)

    def __getitem__(self, idx):

        if self.set_name == 'train' or self.set_name == 'test':

            data_json = self.dataset_json[idx]
            
            name_fp = data_json["name"]

            target_tensor, atten_mask, padding_mask, cond_boundary, cond_door, cond_number, cond_category, cond_partial = get_datatensor(self.set_name, data_json, self.max_num_rooms, self.canvas_size, self.support_boundary, self.support_conditions, self.support_partial)

            cond = {
                'atten_mask': atten_mask,
                'padding_mask': 1-padding_mask,
                'cond_boundary': cond_boundary,
                'cond_door': cond_door,
                'cond_number': cond_number,
                'cond_category': cond_category,
                'cond_partial': cond_partial
            }
            
        if self.set_name == 'test':

            cond.update({
                'syn_atten_mask': atten_mask,
                'syn_padding_mask': 1-padding_mask,
                'syn_cond_boundary': cond_boundary,
                'syn_cond_door': cond_door,
                'syn_cond_number': cond_number,
                'syn_cond_category': cond_category,
                'syn_cond_partial': cond_partial,
                'name_fp': name_fp
            })

        target_tensor = np.transpose(target_tensor, [1, 0]).astype(float)
        
        return target_tensor, cond


if __name__ == '__main__':

    dataset = RPlanDataset('test', True, '', False)
    dataset.__getitem__(500)
