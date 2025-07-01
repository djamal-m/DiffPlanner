import numpy as np
import cv2
import copy
import PIL.Image as Image
import os
import json


color_map = np.array([
    [244, 242, 229],     # living room 0
    [251, 154, 153],     # bedroom  1
    [178, 223, 128],     # kitchen 2
    [166, 206, 227],     # bathroom 3
    [253, 191, 111],     # balcony 4
    [202, 178, 214],     # storage 5
    [57,  67,  94 ],     # exterior wall 6
    [217, 221, 229],     # entrance 7
    [255, 255, 255],     # interior wall 8
    [255, 255, 255]      # external 9
], dtype=np.int64)


pi = 3.141592653589793


def draw_window(mask, windows):

    maskcopy = copy.deepcopy(mask)
    canvas_size = maskcopy.shape[0]
    canvas_size_k = canvas_size // 256
    pixel_len = canvas_size // 80
    node_len = pixel_len // 2

    thickness = node_len

    for k in range(len(windows)):
        window = windows[k]
        seg = np.zeros((2,2))
        seg[0] = window[1:3]
        seg[1] = window[1:3]+ window[3:5]
        
        seg = np.array(seg, dtype=np.int32) * canvas_size_k
        box = np.concatenate([seg.min(0),seg.max(0)],axis=-1)

        if window[3] < window[4]:
            box = box + np.array([-1,0,1,0]) * thickness
            if window[4] > 0:
                box[1] = box[1] + thickness
                seg[0,1] = seg[0,1] + thickness
            else:
                box[3] = box[3] - thickness
                seg[0,1] = seg[0,1] - thickness
        else:
            box = box + np.array([0,-1,0,1]) * thickness
            if window[3] > 0:
                box[0] = box[0] + thickness
                seg[0,0] = seg[0,0] + thickness
            else:
                box[2] = box[2] - thickness
                seg[0,0] = seg[0,0] - thickness
        
        # Draw rectangle in OpenCV
        cv2.rectangle(maskcopy, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)  # Fill white

        # Draw outline and line
        cv2.rectangle(maskcopy, (box[0], box[1]), (box[2], box[3]), color_map[6].tolist(), 1)  # Outline
        cv2.line(maskcopy, (seg[0][0], seg[0][1]), (seg[1][0], seg[1][1]), color_map[6].tolist(), 1)  # Diagonal line

    return maskcopy


def draw_door(mask, doors):

    maskcopy = copy.deepcopy(mask)
    canvas_size = maskcopy.shape[0]
    canvas_size_k = canvas_size // 256
    pixel_len = canvas_size // 80
    node_len = pixel_len // 2

    thickness = node_len

    for k in range(len(doors)):
        door = doors[k]
        seg = np.zeros((2, 2))
        seg[0] = door[1:3]
        seg[1] = door[1:3] + door[3:5]

        seg = np.array(seg, dtype=np.int32) * canvas_size_k

        box = np.concatenate([seg.min(0), seg.max(0)], axis=-1)

        if door[3] < door[4]:
            box += np.array([-thickness, 0, thickness, 0])
            if door[4] > 0:
                box[1] += thickness
            else:
                box[3] -= thickness
        else:
            box += np.array([0, -thickness, 0, thickness])
            if door[3] > 0:
                box[0] += thickness
            else:
                box[2] -= thickness

        # Draw rectangle in OpenCV
        # Note: OpenCV's rectangle function expects top-left and bottom-right as input
        cv2.rectangle(maskcopy, (box[0], box[1]), (box[2], box[3]), color_map[7].tolist(), -1)  # -1 fills the rectangle
        cv2.rectangle(maskcopy, (box[0], box[1]), (box[2], box[3]), color_map[9].tolist(), 1)  # Outline

    return maskcopy


def draw_floorplan(boundary_ini, boxes, types, rdoors=[], rwindows=[], canvas_size=256, draw_entrance=True):

    canvas_size_k = canvas_size // 256
    pixel_len = canvas_size // 80
    node_len = pixel_len // 2

    resultImg = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    resultImg[:, :] = color_map[9].tolist()

    inside_mask_3c = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    boundary = copy.deepcopy(boundary_ini[:,:2])
    boundary = np.concatenate([boundary, boundary[:1]])
    boundary =  np.array(boundary, dtype=np.int32) * canvas_size_k
    cv2.fillPoly(inside_mask_3c, boundary.reshape(1, -1, 2), [1, 1, 1])
    
    liv_color = color_map[0].tolist()
    cv2.fillPoly(resultImg, boundary.reshape(1, -1, 2), liv_color)

    r_boxes = copy.deepcopy(boxes)
    r_types = copy.deepcopy(types)
    
    for i in range(len(r_types)):
        t = r_types[i]
        if t == 0: continue
        min_x = r_boxes[i][0]
        min_y = r_boxes[i][1]
        max_x = r_boxes[i][2]
        max_y = r_boxes[i][3]

        coords_rbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ]

        pos_target = np.array(coords_rbox, dtype=np.int32) * canvas_size_k
        room_color = color_map[t].tolist()
        cv2.fillPoly(resultImg, pos_target.reshape(1, -1, 2), color=room_color)
        inwall_color = color_map[8].tolist()
        cv2.polylines(resultImg, pos_target.reshape(1, -1, 2), isClosed=True, color=inwall_color, thickness=pixel_len)

        node_len = pixel_len // 2
        for pos in pos_target:
            resultImg[pos[1] - node_len:pos[1] + node_len+1, pos[0] - node_len:pos[0] + node_len+1] = inwall_color
    
    resultImg[inside_mask_3c == 0] = 255

    #boundary
    boundary_color = color_map[6].tolist()
    cv2.polylines(resultImg, boundary.reshape(1, -1, 2), isClosed = True, color=boundary_color, thickness = pixel_len)
    node_len = pixel_len // 2
    for pos in boundary:
        resultImg[pos[1] - node_len:pos[1] + node_len+1, pos[0] - node_len:pos[0] + node_len+1] = boundary_color

    #entrance
    if draw_entrance:
        entrance = copy.deepcopy(boundary_ini[:2,:2])
        door_pos = np.array(entrance, dtype=np.int32) * canvas_size_k
        color = color_map[7].tolist()
        cv2.line(resultImg, tuple(door_pos[0]), tuple(door_pos[1]), color, pixel_len)
        node_len = pixel_len // 2
        for pos in door_pos:
            resultImg[pos[1] - node_len:pos[1] + node_len+1, pos[0] - node_len:pos[0] + node_len+1] = color

    if len(rdoors) > 0:
        resultImg = draw_door(resultImg, rdoors)

    if len(rwindows) > 0:
        resultImg = draw_window(resultImg, rwindows)

    return resultImg


def get_realindex(rooms_info, index):
    return next((i for i, room in enumerate(rooms_info) if room["id"] == index), None)


def draw_bubble(boundary_ini=[], rooms_info=[], adjancenies=[], canvas_size=256):

    canvas_size_k = canvas_size // 256
    pixel_len = canvas_size // 80
    node_len = pixel_len // 2

    resultImg = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    resultImg[:, :] = color_map[9].tolist()
    
    if len(boundary_ini)>0:
        #boundary
        boundary_ini = np.array(boundary_ini)
        boundary = copy.deepcopy(boundary_ini[:,:2])
        boundary = np.array(boundary, dtype=np.int32) * canvas_size_k
        boundary_color = color_map[6].tolist()
        cv2.polylines(resultImg, boundary.reshape(1, -1, 2), isClosed=True, color=boundary_color, thickness=pixel_len)
        node_len = pixel_len // 2
        for pos in boundary:
            resultImg[pos[1]-node_len:pos[1]+node_len+1, pos[0]-node_len:pos[0]+node_len+1] = boundary_color

        #entrance
        entrance = copy.deepcopy(boundary_ini[:2,:2])
        door_pos = np.array(entrance, dtype=np.int32) * canvas_size_k
        color = color_map[7].tolist()
        cv2.line(resultImg, tuple(door_pos[0]), tuple(door_pos[1]), color, pixel_len)
        node_len = pixel_len // 2
        for pos in door_pos:
            resultImg[pos[1]-node_len:pos[1]+node_len+1, pos[0]-node_len:pos[0]+node_len+1] = color

    if len(rooms_info) == 0:
        return resultImg

    if len(adjancenies) > 0:
        adjancenies = copy.deepcopy(adjancenies)

        for con in adjancenies:
            start_i = get_realindex(rooms_info, con[0])
            end_i = get_realindex(rooms_info, con[1])

            pos1_512 = [(pos)*canvas_size_k for pos in rooms_info[start_i]['location']]
            pos2_512 = [(pos)*canvas_size_k for pos in rooms_info[end_i]['location']]

            color = color_map[7].tolist()
            cv2.line(resultImg, tuple([pos1_512[0], pos1_512[1]]), tuple([pos2_512[0], pos2_512[1]]), color, pixel_len//2)

    if len(rooms_info) > 0:
        rooms_info = copy.deepcopy(rooms_info)
        for one_room in rooms_info:
            pos_256 = one_room['location']
            pos_512 = [pos*canvas_size_k for pos in pos_256]
            color = color_map[one_room['category']].tolist()
            r_512 = int(pow(one_room['size']*canvas_size_k*canvas_size_k/(pi*8), 0.5))
            cv2.circle(resultImg, tuple([pos_512[0], pos_512[1]]), r_512, color, -1)

    return resultImg


def vis_floorplan(data_json, is_syn=False, support_boundary=True, canvas_size=512, dtype=np.int32):

    data_json = copy.deepcopy(data_json)

    if "boundary" in data_json:
        boundary = data_json["boundary"]
    else:
        boundary = data_json["boundary_aligned"]

    boundary = np.array(boundary).astype(dtype)

    boxes = []
    types = []
    order = []

    rooms_info = data_json["rooms"]

    for room in rooms_info:
        r_c = room["category"]
        if is_syn:
            r_box = room["box_aligned"]
            r_order = room["order_aligned"]
        else:
            r_box = room["box"]
            r_order = room["order"]
        types.append(r_c)
        boxes.append(r_box)
        order.append(r_order)

    types = np.array(types).astype(dtype)
    boxes = np.array(boxes).astype(dtype)
    order = np.array(order).astype(dtype)

    types = types[order]
    boxes = boxes[order]
    
    doors = []
    windows = []
    if "doors" in data_json and "windows" in data_json:
        doors = data_json["doors"]
        windows = data_json["windows"]

    doors = np.array(doors).astype(dtype)
    windows = np.array(windows).astype(dtype)

    resultImg = draw_floorplan(boundary, boxes, types, [], [], canvas_size, support_boundary)
    #resultImg = draw_floorplan(boundary, boxes, types, doors, windows, canvas_size, support_boundary)

    return resultImg


def vis(dataset_path, output_dir, is_syn=False, support_boundary=True):

    syn_prefix = 'syn_' if is_syn else 'gt_'
    boun_prefix = 'wboun_' if support_boundary else 'woboun_'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fp_output_dir = f'{output_dir}/{syn_prefix}{boun_prefix}floorplan'
    if not os.path.exists(fp_output_dir):
        os.mkdir(fp_output_dir)

    bubble_output_dir = f'{output_dir}/{syn_prefix}{boun_prefix}bubble'
    if not os.path.exists(bubble_output_dir):
        os.mkdir(bubble_output_dir)
    
    node_output_dir = f'{output_dir}/{syn_prefix}{boun_prefix}node'
    if not os.path.exists(node_output_dir):
        os.mkdir(node_output_dir)

    with open(dataset_path, encoding='utf-8') as f:
        dataset_json = json.load(f)
    
    num_test = 0
    for data_json in dataset_json:

        name = data_json["name"]
        
        if support_boundary:
            boundary = data_json["boundary"]
        else:
            boundary = []

        node_resultImg = draw_bubble(boundary, data_json["rooms"], [], canvas_size=512)
        output_img = Image.fromarray(np.uint8(node_resultImg))
        output_img.save(f'{node_output_dir}/{name}.png')

        bubble_resultImg = draw_bubble(boundary, data_json["rooms"], data_json["adjacencies"], canvas_size=512)
        output_img = Image.fromarray(np.uint8(bubble_resultImg))
        output_img.save(f'{bubble_output_dir}/{name}.png')

        fp_resultImg = vis_floorplan(data_json, is_syn, support_boundary, canvas_size=512)
        output_img = Image.fromarray(np.uint8(fp_resultImg))
        output_img.save(f'{fp_output_dir}/{name}.png')

        num_test += 1
        if num_test % 1000 == 0:
            print(num_test)

    return 0


if __name__ == '__main__':

    dataset_path = '../dataset/dataset_json/data_test.json'
    output_dir = 'output_vis'
    vis(dataset_path, output_dir, is_syn=False, support_boundary=True)

    # dataset_path = 'output_json/b.json'
    # output_dir = 'output_vis'
    # vis(dataset_path, output_dir, is_syn=True, support_boundary=True)

    # dataset_path = 'output_json/_.json'
    # output_dir = 'output_vis'
    # vis(dataset_path, output_dir, is_syn=True, support_boundary=False)
