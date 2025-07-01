import scipy.io as sio
import numpy as np
import copy
import json
from shapely import geometry as gm


cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1


def find_longest_edge(polygon):

    max_length = 0
    index_of_longest = -1
    num_points = len(polygon)

    for i in range(num_points):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % num_points]
        length = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if length > max_length:
            max_length = length
            index_of_longest = i

    return index_of_longest


def expand_polygon(polygon, target_size=40):

    polygon = copy.deepcopy(polygon)

    while len(polygon) < target_size:
        index_of_longest = find_longest_edge(polygon)
        p1 = polygon[index_of_longest]
        p2 = polygon[(index_of_longest + 1) % len(polygon)]
        
        mid_point = [int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2))]
        polygon.insert(index_of_longest + 1, mid_point)

    return polygon


def get_datajson_from_mat(data_mat):

    data_json = {}

    data_json["name"] = data_mat.name
    data_json["boundary"] = np.array(data_mat.boundary).tolist()

    # boundary_expand
    boundary_expand = data_mat.boundary[:,:2]
    boundary_expand = np.concatenate([boundary_expand, boundary_expand[:1]])
    if not np.all((np.array(boundary_expand) > 0) & (np.array(boundary_expand) < 256)):
        return None
    
    max_corners_boundary = 40
    if len(boundary_expand.tolist()) > max_corners_boundary:
        return None
    
    boundary_expand = expand_polygon(boundary_expand.tolist(), max_corners_boundary)
    data_json["boundary_expand"] = np.array(boundary_expand).tolist()

    # entrance
    entrance = data_mat.boundary[:2,:2]
    entrance_width = 8

    if entrance[0][0] == entrance[1][0]:
        # entrance_ori = 1
        min_x = int(entrance[0][0] - entrance_width)
        max_x = int(entrance[0][0] + entrance_width)
        min_y = int(np.min(entrance[:, 1]))
        max_y = int(np.max(entrance[:, 1]))
    else:
        # entrance_ori = 0
        min_x = int(np.min(entrance[:, 0]))
        max_x = int(np.max(entrance[:, 0]))
        min_y = int(entrance[0][1] - entrance_width)
        max_y = int(entrance[0][1] + entrance_width)

    entrance_expand = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        ]
    
    data_json["entrance_expand"] = entrance_expand

    # rooms info
    rooms_info = []

    rType = data_mat.rType
    rBox = data_mat.gtBoxNew
    rBoundary = data_mat.rBoundary
    order = data_mat.order - 1
    
    boun_polygon = gm.Polygon(data_json["boundary_expand"])
    inside_area = int(boun_polygon.area)
    rooms_area = 0

    for r_i in range(len(rType)):

        r_box = rBox[r_i]
        if not np.all((np.array(r_box) > 0) & (np.array(r_box) < 256)):
            return None

        center_x = int((r_box[0] + r_box[2]) / 2)
        center_y = int((r_box[1] + r_box[3]) / 2)
        r_location = [center_x, center_y]

        width = r_box[2] - r_box[0]
        height = r_box[3] - r_box[1]
        r_size = int(width * height)

        r_category = rType[r_i]
        r_category = int(cIdx[r_category])

        room_info = {}
        room_info["id"] = r_i
        room_info["category"] = r_category
        room_info["size"] = r_size
        room_info["location"] = r_location
        
        room_info["box"] = np.array(r_box).tolist()

        if not np.all((np.array(rBoundary[r_i]) > 0) & (np.array(rBoundary[r_i]) < 256)):
            return None
        r_boundary = np.array(rBoundary[r_i], dtype=np.int32).tolist()
        if len(r_boundary) < 4:
            return None
        
        room_polygon = gm.Polygon(r_boundary)

        room_centroid = room_polygon.centroid.coords[:]
        if len(room_centroid) == 0:
            return None
        room_centroid = room_centroid[0]
        
        room_area = room_polygon.area
        if room_area < 150:
            return None

        room_info["r_boundary"] = r_boundary

        room_info["order"] = int(order[r_i])

        rooms_area += int(room_area)

        rooms_info.append(room_info)
    
    if inside_area != rooms_area:
        return None
    
    data_json["rooms"] = rooms_info

    # adjacencies
    rEdge = data_mat.rEdge
    adjacencies = np.array(rEdge[:, :2]).tolist()
    data_json["adjacencies"] = adjacencies

    # # doors & windows
    # doors = np.array(data_mat.door).tolist()
    # data_json["doors"] = doors
    # windows = np.array(data_mat.window).tolist()
    # data_json["windows"] = windows
    
    return data_json


def main(dataset_mat_path, dataset_json_path):

    dataset_mat = sio.loadmat(dataset_mat_path, squeeze_me=True, struct_as_record=False)['data']
    print(len(dataset_mat))

    dataset_json = []
    for i in range(len(dataset_mat)):

        data_dict = get_datajson_from_mat(dataset_mat[i])
        if data_dict is None:
            continue

        dataset_json.append(data_dict)

        if len(dataset_json) % 10000 == 0:
            print(len(dataset_json))
    
    print(len(dataset_json))
    openw = open(dataset_json_path, 'w')
    json.dump(dataset_json, openw, ensure_ascii=False)
    openw.close()

    return 0


if __name__ == '__main__':

    dataset_mat_path = 'dataset_mat/data_train.mat'
    dataset_json_path = 'dataset_json/data_train.json'
    main(dataset_mat_path, dataset_json_path)