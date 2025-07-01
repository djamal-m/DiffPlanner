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
    support_partial,
    syn_dataset_path=''
):

    deterministic = False if set_name =='train' else True
    dataset = RPlanDataset(set_name, support_boundary, support_conditions, support_partial, syn_dataset_path)

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


def get_realindex(rooms_info, index):
    return next((i for i, room in enumerate(rooms_info) if room["id"] == index), None)


def get_datatensor(set_name, data_json, max_num_rooms, canvas_size, support_boundary, support_conditions, support_partial):

    data_json = copy.deepcopy(data_json)

    if support_boundary:
        boundary_rplan = data_json["boundary_expand"]
        cond_boundary = (np.array(boundary_rplan).flatten().reshape(1, -1) / canvas_size * 2) - 1
        cond_boundary = np.repeat(cond_boundary, max_num_rooms, axis=0)

        door_rplan = data_json["entrance_expand"]
        cond_door = (np.array(door_rplan).flatten().reshape(1, -1) / canvas_size * 2 ) - 1
        cond_door = np.repeat(cond_door, max_num_rooms, axis=0)

    else:
        cond_boundary = np.full((max_num_rooms, 80), -1.0)
        cond_door = np.full((max_num_rooms, 8), -1.0)

    rooms_info = data_json["rooms"]
    
    num_rooms = len(rooms_info)
    
    np.random.seed(None)
    np.random.shuffle(rooms_info)

    canvas_area = canvas_size * canvas_size

    # condition number
    cond_number = np.zeros((max_num_rooms, max_num_rooms))
    for r_i in range(num_rooms): 
        cond_number[r_i] = np.array([get_one_hot(r_i, max_num_rooms)])
    
    # condition node
    cond_node = np.full((max_num_rooms, 4), -1.0)
    for r_i in range(num_rooms):
        csxy_rooms = [(int(rooms_info[r_i]["category"]) + 1) / 3 - 1, (rooms_info[r_i]["size"] / canvas_area * 2) - 1, (rooms_info[r_i]["location"][0] / canvas_size * 2) - 1, (rooms_info[r_i]["location"][1] / canvas_size * 2) - 1]
        cond_node[r_i] = np.array(csxy_rooms)

    # condition adjacency
    adjacencies = data_json["adjacencies"]
    np.random.shuffle(adjacencies)
    
    target_tensor = np.full((max_num_rooms, max_num_rooms), -1.0)
    for adja in adjacencies:
        start_i = get_realindex(rooms_info, adja[0])
        end_i = get_realindex(rooms_info, adja[1])
        target_tensor[start_i, end_i] = 1
        target_tensor[end_i, start_i] = 1

    atten_mask = np.ones((max_num_rooms, max_num_rooms))
    atten_mask[:num_rooms, :num_rooms] = 0

    padding_mask = np.repeat(1, max_num_rooms)
    padding_mask[num_rooms:max_num_rooms] = 0
    
    # condition partial
    cond_partial = np.full((max_num_rooms, max_num_rooms), -1.0)
    if support_partial:

        random_k = random.randint(0, len(adjacencies)-1)

        if set_name == 'test':
            ratio_partial = 0.25
            #ratio_partial = 0.5
            #ratio_partial = 0.75
            random_k = round(len(adjacencies) * ratio_partial)

        if random_k > 0:
            partial_adjacencies = adjacencies[:random_k]
            for adja in partial_adjacencies:
                start_i = get_realindex(rooms_info, adja[0])
                end_i = get_realindex(rooms_info, adja[1])
                cond_partial[start_i, end_i] = 1
                cond_partial[end_i, start_i] = 1

    return target_tensor, atten_mask, padding_mask, cond_boundary, cond_door, cond_number, cond_node, cond_partial


def get_syn_datatensor(data_json, max_num_rooms, canvas_size):

    data_json = copy.deepcopy(data_json)

    rooms_info = data_json["rooms"]
    
    num_rooms = len(rooms_info)
    
    np.random.seed(None)
    np.random.shuffle(rooms_info)

    canvas_area = canvas_size * canvas_size
    
    # condition number
    cond_number = np.zeros((max_num_rooms, max_num_rooms))
    for r_i in range(num_rooms): 
        cond_number[r_i] = np.array([get_one_hot(r_i, max_num_rooms)])
    
    # condition node
    cond_node = np.full((max_num_rooms, 4), -1.0)
    for r_i in range(num_rooms):
        csxy_rooms = [(int(rooms_info[r_i]["category"]) + 1) / 3 - 1, (rooms_info[r_i]["size"] / canvas_area * 2) - 1, (rooms_info[r_i]["location"][0] / canvas_size * 2) - 1, (rooms_info[r_i]["location"][1] / canvas_size * 2) - 1]
        cond_node[r_i] = np.array(csxy_rooms)

    atten_mask = np.ones((max_num_rooms, max_num_rooms))
    atten_mask[:num_rooms, :num_rooms] = 0

    padding_mask = np.repeat(1, max_num_rooms)
    padding_mask[num_rooms:max_num_rooms] = 0
    
    return atten_mask, padding_mask, cond_number, cond_node


class RPlanDataset(Dataset):

    def __init__(self, set_name, support_boundary, support_conditions, support_partial, syn_dataset_path):

        super().__init__()

        self.set_name = set_name
        self.max_num_rooms = 8
        self.canvas_size = 256
        
        self.syn_dataset_path = syn_dataset_path
       
        self.support_boundary = support_boundary # true or false
        self.support_conditions = support_conditions # 'ncsl'
        self.support_partial = support_partial # true or false

        dataset_json_path = f'../../dataset/dataset_json/data_{self.set_name}.json'
        with open(dataset_json_path, encoding='utf-8') as f:
            self.dataset_json = json.load(f)

        if self.set_name == 'test':
            
            if self.syn_dataset_path != '':
                with open(self.syn_dataset_path, encoding='utf-8') as f:
                    self.syn_dataset_json = json.load(f)

                self.name2index_dict = {}
                for data_i in range(len(self.syn_dataset_json)):
                    name = self.syn_dataset_json[data_i]["name"]
                    self.name2index_dict[name] = data_i

    def __len__(self):
        return len(self.dataset_json)

    def __getitem__(self, idx):

        if self.set_name == 'train' or self.set_name == 'test':

            data_json = self.dataset_json[idx]
            
            name_fp = data_json["name"]

            target_tensor, atten_mask, padding_mask, cond_boundary, cond_door, cond_number, cond_node, cond_partial = get_datatensor(self.set_name, data_json, self.max_num_rooms, self.canvas_size, self.support_boundary, self.support_conditions, self.support_partial)

            cond = {
                'atten_mask': atten_mask,
                'padding_mask': 1-padding_mask,
                'cond_boundary': cond_boundary,
                'cond_door': cond_door,
                'cond_number': cond_number,
                'cond_node': cond_node,
                'cond_partial': cond_partial 
            }
            
        if self.set_name == 'test':
            
            if self.syn_dataset_path != '':
                syn_data_json = self.syn_dataset_json[self.name2index_dict[name_fp]]
                atten_mask, padding_mask, cond_number, cond_node = get_syn_datatensor(syn_data_json, self.max_num_rooms, self.canvas_size)

            cond.update({
                'syn_atten_mask': atten_mask,
                'syn_padding_mask': 1-padding_mask,
                'syn_cond_boundary': cond_boundary,
                'syn_cond_door': cond_door,
                'syn_cond_number': cond_number,
                'syn_cond_node': cond_node,
                'syn_cond_partial': cond_partial,
                'name_fp':name_fp
            })

        target_tensor = np.transpose(target_tensor, [1, 0]).astype(float)
        
        return target_tensor, cond
        
if __name__ == '__main__':
    dataset = RPlanDataset('test', True, 'ncsl', False, '')
    dataset.__getitem__(500)
