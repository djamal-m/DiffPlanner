import numpy as np
import copy
import json
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box as shapely_box
from shapely.ops import unary_union

from decorate import get_dw


def get_boundary_from_boxes(boxes):

    polygons = []
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        polygons.append(rect)

    merged_result = unary_union(polygons)

    if merged_result.geom_type == 'MultiPolygon':
        threshold = 32
        merged_orthogonal = merged_result.buffer(threshold, join_style=2).buffer(-threshold, join_style=2)
    else:
        threshold = 6
        merged_orthogonal = merged_result.buffer(threshold, join_style=2).buffer(-threshold, join_style=2)

    x, y = merged_orthogonal.exterior.xy
    boundary = np.column_stack((x, y))[:-1]

    return boundary


def align_fp_woboundary(rBox, rType, rEdge, threshold=12):
    
    rBox = copy.deepcopy(rBox)
    rType = copy.deepcopy(rType)
    rEdge = copy.deepcopy(rEdge)
    
    newBox = rBox.copy()
    updated = np.zeros_like(newBox, dtype=bool)
    _, newBox, _ = align_neighbor(newBox, rEdge, updated, threshold + 6)
    
    boundary = get_boundary_from_boxes(newBox)

    zeros = np.zeros((boundary.shape[0], 2))
    boundary = np.hstack((boundary, zeros))

    newBox, order = regularize_fp(newBox, boundary, rType)

    newBox, rBoundary = get_room_boundary(newBox, boundary, order)

    return newBox, order-1, rBoundary, boundary


def align_fp(boundary, rBox, rType, rEdge, threshold=12):
    
    boundary = copy.deepcopy(boundary)
    rBox = copy.deepcopy(rBox)
    rType = copy.deepcopy(rType)
    rEdge = copy.deepcopy(rEdge)

    living_indices = np.where(rType == 0)[0]
    if living_indices.size == 0:
        raise ValueError("no living room")
    
    idx = np.isin(rEdge[:, 0], living_indices) | np.isin(rEdge[:, 1], living_indices)

    rEdge_filtered = rEdge[~idx, :]
    entranceBox = get_entrance_space(boundary[0:2, 0:2], boundary[0, 2], threshold)

    _, newBox, updated = align_with_boundary(rBox, boundary, threshold, rType)

    _, newBox, _ = align_neighbor(newBox, rEdge_filtered, updated, threshold + 6)
    
    newBox, order = regularize_fp(newBox, boundary, rType)

    newBox, rBoundary = get_room_boundary(newBox, boundary, order)

    return newBox, order-1, rBoundary


def get_entrance_space(doorSeg, doorOri, threshold):

    doorBox = [doorSeg[0, 0], doorSeg[0, 1], doorSeg[1, 0], doorSeg[1, 1]]

    if doorOri == 0:
        doorBox[3] += threshold
    elif doorOri == 1:
        doorBox[0] -= threshold
    elif doorOri == 2:
        doorBox[1] -= threshold
    elif doorOri == 3:
        doorBox[2] += threshold
    
    minx, maxx = min(doorBox[0], doorBox[2]), max(doorBox[0], doorBox[2])
    miny, maxy = min(doorBox[1], doorBox[3]), max(doorBox[1], doorBox[3])

    return [minx, miny, maxx, maxy]


def align_with_boundary(box, boundary, threshold, rType):

    tempBox = box.copy()
    updated = np.zeros_like(box, dtype=bool)
    closedSeg = np.zeros_like(box, dtype=box.dtype)
    distSeg = np.zeros_like(box, dtype=float)

    for i in range(len(box)):
        closedSeg_i, distSeg_i, idx_i = find_close_seg(box[i], boundary)
        closedSeg[i, :] = closedSeg_i
        distSeg[i, :] = distSeg_i

    mask = distSeg <= threshold
    box[mask] = closedSeg[mask]
    updated[mask] = True

    idx = np.where(np.any(mask, axis=1))[0]

    constraint = []
    for i in idx:
        constraint.append([i] + closedSeg[i].tolist())
    constraint = np.array(constraint)

    entranceBox = get_entrance_space(boundary[0:2, 0:2], boundary[0, 2], threshold)

    if len(entranceBox) >= 4:
        entrancePoly = shapely_box(entranceBox[0], entranceBox[1], entranceBox[2], entranceBox[3])
        x_entranceBox = [0, entranceBox[1], 255, entranceBox[3]]
        y_entranceBox = [entranceBox[0], 255, entranceBox[2], 0]

        x_entrancePoly = shapely_box(x_entranceBox[0], x_entranceBox[1], x_entranceBox[2], x_entranceBox[3])
        y_entrancePoly = shapely_box(y_entranceBox[0], y_entranceBox[1], y_entranceBox[2], y_entranceBox[3])
    else:
        entrancePoly = Polygon()
        x_entrancePoly = Polygon()
        y_entrancePoly = Polygon()
    
    liv_idxs = np.where(rType == 0)[0]
    if len(liv_idxs) > 0:
        liv_idx = liv_idxs[0]
        livPoly = shapely_box(box[liv_idx, 0], box[liv_idx, 1], box[liv_idx, 2], box[liv_idx, 3])
    else:
        livPoly = Polygon()

    shrink_ori = boundary[0, 2]
    if shrink_ori % 2 == 0:
        if y_entrancePoly.is_valid and livPoly.is_valid and y_entrancePoly.intersects(livPoly):
            shrink_ori = 4
        elif x_entrancePoly.is_valid and livPoly.is_valid and x_entrancePoly.intersects(livPoly):
            shrink_ori = 5
    else:
        if x_entrancePoly.is_valid and livPoly.is_valid and x_entrancePoly.intersects(livPoly):
            shrink_ori = 5
        elif y_entrancePoly.is_valid and livPoly.is_valid and y_entrancePoly.intersects(livPoly):
            shrink_ori = 4

    for i in range(len(box)):
        if rType[i] != 0:
            roomPoly = shapely_box(box[i, 0], box[i, 1], box[i, 2], box[i, 3])

            if entrancePoly.is_valid and roomPoly.is_valid and entrancePoly.intersects(roomPoly):
                
                new_room_poly = shrink_box(roomPoly, entrancePoly, shrink_ori)
                
                box[i, :] = np.array([new_room_poly[0], new_room_poly[1],
                                      new_room_poly[2], new_room_poly[3]]).astype(box.dtype)
                
                updated[i, :] = box[i, :] != tempBox[i, :]
                
    return constraint, box, updated


def align_neighbor(box, rEdge, updated, threshold):
    
    if updated.size == 0:
        updated = np.zeros_like(box, dtype=bool)
    
    tempBox = box.copy()
    constraint = []
    checked = np.zeros(rEdge.shape[0], dtype=bool)
    updatedCount = get_updated_count(updated, rEdge)
    
    for _ in range(rEdge.shape[0]):
        I = np.where(~checked)[0]
        if I.size == 0:
            break
        
        t = I[np.argmax(updatedCount[I])]
        checked[t] = True
        
        idx_rooms = rEdge[t, 0:2].astype(int)
        idx_rooms = np.clip(idx_rooms, 0, box.shape[0]-1)
        
        b, c = align_adjacent_room(box[idx_rooms, :], tempBox[idx_rooms, :], updated[idx_rooms, :], rEdge[t, 2], threshold)
        
        for j in range(len(idx_rooms)):
            if j >= len(c):
                continue
            if not isinstance(c[j], (list, tuple)):
                continue
            for constraint_idx in c[j]:
                if constraint_idx >= box.shape[1]:
                    continue
                updated[idx_rooms[j], constraint_idx] = True
                constraint.append([idx_rooms[j], constraint_idx])
            
            if b[j, 0] == box[idx_rooms[j], 0]:
                updated[idx_rooms[j], 0] = False
            if b[j, 1] == box[idx_rooms[j], 1]:
                updated[idx_rooms[j], 1] = False
            if b[j, 2] == box[idx_rooms[j], 2]:
                updated[idx_rooms[j], 2] = False
            if b[j, 3] == box[idx_rooms[j], 3]:
                updated[idx_rooms[j], 3] = False
        
        box[idx_rooms, :] = b
        
        updatedCount = get_updated_count(updated, rEdge)
    
    constraint = np.array(constraint)
    
    return constraint, box, updated


def get_updated_count(updated, rEdge):

    K = rEdge.shape[0]
    updatedCount = np.zeros(K, dtype=int)
    for k in range(K):
        rooms = rEdge[k, 0:2].astype(int)
        updatedCount[k] = np.sum(updated[rooms, :])

    return updatedCount


def align_adjacent_room(box, tempBox, updated, type, threshold):

    newBox = box.copy()
    constraint = []

    def alignV(isLeft):
        if isLeft:
            idx1 = [1, 0]  # [2,1] → box2[x1]
            idx2 = [0, 2]  # [1,3] → box1[x2]
        else:
            idx1 = [1, 2]  # [2,3] → box2[x2]
            idx2 = [0, 0]  # [1,1] → box1[x1]

        cond = abs(tempBox[idx1[0], idx1[1]] - tempBox[idx2[0], idx2[1]]) <= abs(tempBox[idx1[0], idx2[1]] - tempBox[idx2[0], idx2[1]])

        if cond:
            align(idx1, idx2, threshold / 2)
        else:
            align([1, 2], [0, 2], threshold / 2)

    def alignH(isAbove):
        if isAbove:
            idx1 = [1, 1]  # [2,2] → box2[y1]
            idx2 = [0, 3]  # [1,4] → box1[y2]
        else:
            idx1 = [1, 3]  # [2,4] → box2[y2]
            idx2 = [0, 1]  # [1,2] → box1[y1]

        cond = abs(tempBox[idx1[0], idx1[1]] - tempBox[idx2[0], idx2[1]]) <= abs(tempBox[idx1[0], idx2[1]] - tempBox[idx2[0], idx2[1]])

        if cond:
            align(idx1, idx2, threshold / 2)
        else:
            align([1, 3], [0, 3], threshold / 2)

    def align(idx1, idx2, threshold, attach=False):
        difference = abs(tempBox[idx1[0], idx1[1]] - tempBox[idx2[0], idx2[1]])

        if difference <= threshold:
            if updated[idx1[0], idx1[1]] and not updated[idx2[0], idx2[1]]:
                newBox[idx2[0], idx2[1]] = newBox[idx1[0], idx1[1]]
            elif updated[idx2[0], idx2[1]] and not updated[idx1[0], idx1[1]]:
                newBox[idx1[0], idx1[1]] = newBox[idx2[0], idx2[1]]
            elif not updated[idx1[0], idx1[1]] and not updated[idx2[0], idx2[1]]:
                if attach:
                    newBox[idx2[0], idx2[1]] = newBox[idx1[0], idx1[1]]
                else:
                    y = (newBox[idx1[0], idx1[1]] + newBox[idx2[0], idx2[1]]) / 2
                    newBox[idx1[0], idx1[1]] = y
                    newBox[idx2[0], idx2[1]] = y

            if idx1[0] == 0:  # box1
                constraint.append([idx1[1], idx2[1]])
            else:  # box2
                constraint.append([idx2[1], idx1[1]])

    if type == 0:
        alignV(True)
        alignH(True)
    elif type == 1:
        alignV(True)
        alignH(False)
    elif type == 2:
        align([1, 0], [0, 2], threshold)
        align([1, 1], [0, 1], threshold / 2)
        align([1, 3], [0, 3], threshold / 2)
    elif type == 3:
        align([1, 1], [0, 3], threshold)
        align([1, 0], [0, 0], threshold / 2)
        align([1, 2], [0, 2], threshold / 2)
    elif type == 4:
        align([1, 0], [0, 0], threshold, attach=True)
        align([1, 1], [0, 1], threshold, attach=True)
        align([1, 2], [0, 2], threshold, attach=True)
        align([1, 3], [0, 3], threshold, attach=True)
    elif type == 5:
        align([0, 0], [1, 0], threshold, attach=True)
        align([0, 1], [1, 1], threshold, attach=True)
        align([0, 2], [1, 2], threshold, attach=True)
        align([0, 3], [1, 3], threshold, attach=True)
    elif type == 6:
        align([1, 3], [0, 1], threshold)
        align([1, 0], [0, 0], threshold / 2)
        align([1, 2], [0, 2], threshold / 2)
    elif type == 7:
        align([1, 2], [0, 0], threshold)
        align([1, 1], [0, 1], threshold / 2)
        align([1, 3], [0, 3], threshold / 2)
    elif type == 8:
        alignV(False)
        alignH(True)
    elif type == 9:
        alignV(False)
        alignH(False)

    return newBox, constraint


def get_room_boundary(box, boundary, order):
    
    isNew = boundary[:, 3].astype(bool)
    boundary_filtered = boundary[~isNew, :].astype(float)
    
    boundary_coords = boundary_filtered[:, :2]
    
    if not np.array_equal(boundary_coords[0], boundary_coords[-1]):
        boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
    
    polyBoundary = Polygon(boundary_coords)
    
    if not polyBoundary.is_valid:
        polyBoundary = polyBoundary.buffer(0)
        if not polyBoundary.is_valid:
            raise ValueError("polyBoundary is invalid and could not be fixed.")
    
    poly = []
    for i in range(box.shape[0]):
        room_coords = [
            (box[i, 0], box[i, 1]),
            (box[i, 0], box[i, 3]),
            (box[i, 2], box[i, 3]),
            (box[i, 2], box[i, 1])
        ]
        polyRoom = Polygon(room_coords)
        
        if not polyRoom.is_valid:
            polyRoom = polyRoom.buffer(0)
            if not polyRoom.is_valid:
                raise ValueError(f"Room {i+1} polygon is invalid and could not be fixed.")
        
        poly.append(polyRoom)
    
    newBox = box.copy()
    rBoundary = [None] * box.shape[0]
    
    for i in range(len(order)):
        idx = order[i] - 1
        
        rPoly = polyBoundary.intersection(poly[idx])

        if not rPoly.is_valid:
            rPoly = rPoly.buffer(0)
            if not rPoly.is_valid:
                raise ValueError(f"Intersection for room {idx+1} is invalid and could not be fixed.")
        
        for j in range(i + 1, len(order)):
            sub_idx = order[j] - 1
            rPoly = rPoly.difference(poly[sub_idx])
            
            if not rPoly.is_valid:
                rPoly = rPoly.buffer(0)
                if not rPoly.is_valid:
                    raise ValueError(f"Difference operation for room {idx+1} with room {sub_idx+1} resulted in invalid geometry.")
        
        if not rPoly.is_empty:
            if isinstance(rPoly, Polygon):
                rBoundary[idx] = np.array(rPoly.exterior.coords)
            elif isinstance(rPoly, MultiPolygon):
                rBoundary[idx] = np.vstack([np.array(p.exterior.coords) for p in rPoly.geoms])
            elif isinstance(rPoly, GeometryCollection):
                coords = []
                for geom in rPoly.geoms:
                    if isinstance(geom, Polygon):
                        coords.extend(list(geom.exterior.coords))
                if coords:
                    rBoundary[idx] = np.array(coords)
                else:
                    rBoundary[idx] = None
            else:
                rBoundary[idx] = None
            
            x_min, y_min, x_max, y_max = rPoly.bounds
            newBox[idx, :] = [x_min, y_min, x_max, y_max]
        else:
            print(f"Warning: Room {idx+1} has an empty intersection after difference operations.")
    
    return newBox, rBoundary


def find_close_seg(box, boundary):
    
    isNew = boundary[:, 3].astype(bool)
    boundary_filtered = boundary[~isNew].astype(float)

    bSeg = np.hstack((boundary_filtered[:, 0:2],
                      np.roll(boundary_filtered[:, 0:2], -1, axis=0),
                      boundary_filtered[:, 2:3]))
    
    vSeg = bSeg[bSeg[:, 4] % 2 == 1].copy()

    flag_3 = vSeg[:, 4] == 3
    vSeg[flag_3, 1], vSeg[flag_3, 3] = vSeg[flag_3, 3].copy(), vSeg[flag_3, 1].copy()

    vSeg = vSeg[np.argsort(vSeg[:, 0])]

    hSeg = bSeg[bSeg[:, 4] % 2 == 0].copy()

    flag_2 = hSeg[:, 4] == 2
    hSeg[flag_2, 0], hSeg[flag_2, 2] = hSeg[flag_2, 2].copy(), hSeg[flag_2, 0].copy()

    hSeg = hSeg[np.argsort(hSeg[:, 1])]

    closedSeg = np.full(4, 256.0)
    distSeg = np.full(4, 256.0)
    idx = np.zeros(4, dtype=int)

    for i, seg in enumerate(vSeg):
        # seg = [x1, y1, x2, y2, flag]
        vdist = 0.0
        if seg[3] <= box[1]:
            vdist = box[1] - seg[3]
        elif seg[1] >= box[3]:
            vdist = seg[1] - box[3]
        
        hdist = np.array([box[0] - seg[0], box[2] - seg[0]])
        dist1 = np.linalg.norm([hdist[0], vdist])
        dist3 = np.linalg.norm([hdist[1], vdist])

        if dist1 < distSeg[0] and dist1 <= dist3 and hdist[0] > 0:
            distSeg[0] = dist1
            idx[0] = i
            closedSeg[0] = seg[0]
        elif dist3 < distSeg[2] and hdist[1] < 0:
            distSeg[2] = dist3
            idx[2] = i
            closedSeg[2] = seg[2]

    for i, seg in enumerate(hSeg):
        # seg = [x1, y1, x2, y2, flag]
        hdist = 0.0
        if seg[2] <= box[0]:
            hdist = box[0] - seg[2]
        elif seg[0] >= box[2]:
            hdist = seg[0] - box[2]
        
        vdist = np.array([box[1] - seg[1], box[3] - seg[1]])
        dist2 = np.linalg.norm([vdist[0], hdist])
        dist4 = np.linalg.norm([vdist[1], hdist])

        if dist2 <= dist4 and dist2 < distSeg[1] and vdist[0] > 0:
            distSeg[1] = dist2
            idx[1] = i
            closedSeg[1] = seg[1]
        elif dist4 < distSeg[3] and vdist[1] < 0:
            distSeg[3] = dist4
            idx[3] = i
            closedSeg[3] = seg[3]

    return closedSeg, distSeg, idx


def shrink_box(roomPoly, entrancePoly, doorOrient):

    PG = roomPoly.difference(entrancePoly)
    if PG.is_empty:
        return [0, 0, 0, 0]
    x1, y1 = roomPoly.centroid.coords[0]
    x2, y2 = entrancePoly.centroid.coords[0]
    box = [roomPoly.bounds[0], roomPoly.bounds[1], roomPoly.bounds[2], roomPoly.bounds[3]]
    if doorOrient % 2 == 0:  # Door grows vertically
        if x1 < x2:
            box[2] = min(entrancePoly.bounds[0], box[2])
        else:
            box[0] = max(entrancePoly.bounds[2], box[0])
    else:
        if y1 < y2:
            box[3] = min(entrancePoly.bounds[1], box[3])
        else:
            box[1] = max(entrancePoly.bounds[3], box[1])

    return box


def find_room_order(M):
    
    import networkx as nx
    n = M.shape[0]
    G = nx.DiGraph(M)
    
    name_mapping = {i: str(i+1) for i in range(n)}
    G = nx.relabel_nodes(G, name_mapping)
    
    order = []
    remaining_graph = G.copy()
    
    while len(order) < n:
        in_degrees = dict(remaining_graph.in_degree())
        
        zero_indegree_nodes = [node for node, deg in in_degrees.items() if deg == 0]
        
        if zero_indegree_nodes:
            for node in zero_indegree_nodes:
                order.append(int(node))
                remaining_graph.remove_node(node)
        else:
            min_indegree = min(in_degrees.values())
            candidates = [node for node, deg in in_degrees.items() if deg == min_indegree]
            selected_node = candidates[0]
            order.append(int(selected_node))
            remaining_graph.remove_node(selected_node)
    
    return np.array(order)


def regularize_fp(box, boundary, rType):
    
    isNew = boundary[:, 3].astype(bool)
    boundary_filtered = boundary[~isNew, :].astype(float)
    
    boundary_coords = boundary_filtered[:, 0:2]

    if not np.array_equal(boundary_coords[0], boundary_coords[-1]):
        boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
    polyBoundary = Polygon(boundary_coords)
    
    for i in range(box.shape[0]):
        polyRoom = Polygon([
            (box[i, 0], box[i, 1]),
            (box[i, 0], box[i, 3]),
            (box[i, 2], box[i, 3]),
            (box[i, 2], box[i, 1])
        ])
        intersection = polyBoundary.intersection(polyRoom)
        
        if intersection.is_empty:
            print(f'Room {i} is outside the building!')
        else:
            x_min, y_min, x_max, y_max = intersection.bounds
            box[i, :] = [x_min, y_min, x_max, y_max]
    
    M = box.shape[0]
    orderM = np.zeros((M, M), dtype=bool)
    
    polyRooms = [Polygon([
        (box[i, 0], box[i, 1]),
        (box[i, 0], box[i, 3]),
        (box[i, 2], box[i, 3]),
        (box[i, 2], box[i, 1])
    ]) for i in range(M)]
    areas = [polyRooms[i].area for i in range(M)]
    
    for i in range(M):
        for j in range(i+1, M):
            inter = polyRooms[i].intersection(polyRooms[j])
            if not inter.is_empty and inter.area > 0:
                if areas[i] <= areas[j]:
                    orderM[i, j] = True
                else:
                    orderM[j, i] = True
    
    order = np.arange(1, M+1)
    if np.any(orderM):
        order = find_room_order(orderM)
    order = order[::-1]
    
    livingIdx = np.where(rType == 0)[0]
    if livingIdx.size == 0:
        raise ValueError("no living room")
    livingIdx = livingIdx[0]
    
    for i in range(M):
        if i != livingIdx:
            if box[i, 0] == box[i, 2] or box[i, 1] == box[i, 3]:
                print(f'Empty box {i}!!!')
            else:
                polyRoom = Polygon([
                    (box[i, 0], box[i, 1]),
                    (box[i, 0], box[i, 3]),
                    (box[i, 2], box[i, 3]),
                    (box[i, 2], box[i, 1])
                ])
                polyBoundary = polyBoundary.difference(polyRoom)
    
    livingPoly = Polygon([
        (box[livingIdx, 0], box[livingIdx, 1]),
        (box[livingIdx, 0], box[livingIdx, 3]),
        (box[livingIdx, 2], box[livingIdx, 3]),
        (box[livingIdx, 2], box[livingIdx, 1])
    ])
    
    gap = polyBoundary
    if gap.is_empty:
        print("No gaps inside the building.")
    elif isinstance(gap, Polygon):
        x_min, y_min, x_max, y_max = gap.bounds
        box[livingIdx, :] = [x_min, y_min, x_max, y_max]
    elif isinstance(gap, MultiPolygon):
        regions = list(gap.geoms)
        overlapArea = []
        closeRoomIdx = []
        
        for region in regions:
            if region.intersects(livingPoly):
                inter = region.intersection(livingPoly)
                overlapArea.append(inter.area)
            else:
                overlapArea.append(0)
            
            centroid = region.centroid
            center = np.array([centroid.x, centroid.y])
            
            room_centers = (box[:, 0:2] + box[:, 2:4]) / 2
            distances = np.linalg.norm(room_centers - center, axis=1)
            bIdx = np.argmin(distances)
            closeRoomIdx.append(bIdx)
        
        overlapArea = np.array(overlapArea)
        closeRoomIdx = np.array(closeRoomIdx)
        
        if overlapArea.size == 0:
            print("No overlapping regions found.")
            return box, order
        
        lIdx = np.argmax(overlapArea)
        
        for k in range(len(closeRoomIdx)):
            if k == lIdx:
                x_min, y_min, x_max, y_max = regions[k].bounds
                box[livingIdx, :] = [x_min, y_min, x_max, y_max]
            else:
                room_idx = closeRoomIdx[k]
                roomPoly = Polygon([
                    (box[room_idx, 0], box[room_idx, 1]),
                    (box[room_idx, 0], box[room_idx, 3]),
                    (box[room_idx, 2], box[room_idx, 3]),
                    (box[room_idx, 2], box[room_idx, 1])
                ])
                union_poly = roomPoly.union(regions[k])
                x_min, y_min, x_max, y_max = union_poly.bounds
                box[room_idx, :] = [x_min, y_min, x_max, y_max]
    
    return box, order


edge_type = [
    'left-above',
    'left-below',
    'left-of',
    'above',
    'inside',
    'surrounding',
    'below',
    'right-of',
    'right-above',
    'right-below'
]


def collide2d(bbox1, bbox2, th=0):
    return not (
        (bbox1[0] - th > bbox2[2]) or
        (bbox1[2] + th < bbox2[0]) or
        (bbox1[1] - th > bbox2[3]) or
        (bbox1[3] + th < bbox2[1])
    )


def point_box_relation(u, vbox):
    uy, ux = u
    vy0, vx0, vy1, vx1 = vbox
    if (ux < vx0 and uy <= vy0) or (ux == vx0 and uy == vy0):
        relation = 0  # 'left-above'
    elif (vx0 <= ux < vx1 and uy <= vy0):
        relation = 3  # 'above'
    elif (vx1 <= ux and uy < vy0) or (ux == vx1 and uy == vy0):
        relation = 8  # 'right-above'
    elif (vx1 <= ux and vy0 <= uy < vy1):
        relation = 7  # 'right-of'
    elif (vx1 < ux and vy1 <= uy) or (ux == vx1 and uy == vy1):
        relation = 9  # 'right-below'
    elif (vx0 < ux <= vx1 and vy1 <= uy):
        relation = 6  # 'below'
    elif (ux <= vx0 and vy1 < uy) or (ux == vx0 and uy == vy1):
        relation = 1  # 'left-below'
    elif (ux <= vx0 and vy0 < uy <= vy1):
        relation = 2  # 'left-of'
    elif (vx0 < ux < vx1 and vy0 < uy < vy1):
        relation = 4  # 'inside'
    else:
        relation = -1 #

    return relation


def get_adjancenies_from_boxes(boxes, th=9):
    edges = []
    for u in range(len(boxes)):
        for v in range(u + 1, len(boxes)):
            if not collide2d(boxes[u, :4], boxes[v, :4], th=th):
                continue
            uy0, ux0, uy1, ux1 = boxes[u, :4].astype(int)
            vy0, vx0, vy1, vx1 = boxes[v, :4].astype(int)
            uc = ((uy0 + uy1) / 2, (ux0 + ux1) / 2)
            vc = ((vy0 + vy1) / 2, (vx0 + vx1) / 2)
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 5  # 'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 4  # 'inside'
            else:
                relation = point_box_relation(uc, boxes[v, :4])
            if relation != -1:
                edges.append([u, v, relation])

    edges = np.array(edges, dtype=int)

    return edges


def get_datamat_from_json(data_json, support_boundary=True, dtype=np.int32):
    
    data_json = copy.deepcopy(data_json)
    name = data_json["name"]

    if support_boundary:
        boundary = data_json["boundary"]
    else:
        boundary = []

    gen_rBoxs = []
    gen_rTypes = []

    rooms_info = data_json["rooms"]

    for room in rooms_info:
        r_c = room["category"]
        gen_rTypes.append(r_c)
        r_box = np.array(room["box"])
        gen_rBoxs.append(r_box)

    boxes = np.array(gen_rBoxs).astype(dtype)

    boxes_yxyx = (boxes[:,[1,0,3,2]]).astype(int)
    edges = np.array(get_adjancenies_from_boxes(boxes_yxyx), dtype=np.int32)

    return {
        'name': name,
        'boundary': np.array(boundary).astype(dtype),
        'boxes': np.array(boxes).astype(dtype),
        'types': np.array(gen_rTypes).astype(dtype),
        'edges': np.array(edges).astype(dtype)
    }


def get_datajson_aligned_from_mat(data_mat, data_json, support_boundary=True):
    
    data_json_aligned = copy.deepcopy(data_json)

    boxes_aligned = data_mat['boxes_aligned']
    room_boundaries = data_mat['room_boundaries']
    order = data_mat['order']
    edges_aligned = data_mat['edges_aligned']
    
    rooms_info = data_json_aligned["rooms"]
    num_rooms = len(rooms_info)
    for r_i in range(num_rooms):
        r_box = boxes_aligned[r_i]
        center_x = int((r_box[0] + r_box[2]) / 2)
        center_y = int((r_box[1] + r_box[3]) / 2)
        r_location = [center_x, center_y]

        width = r_box[2] - r_box[0]
        height = r_box[3] - r_box[1]
        r_size = int(width * height)

        rooms_info[r_i]["size_aligned"] = r_size
        rooms_info[r_i]["location_aligned"] = r_location
        rooms_info[r_i]["box_aligned"] = np.array(r_box).tolist()

        r_boundary_aligned = np.array(room_boundaries[r_i], dtype=np.int32).tolist()
        rooms_info[r_i]["r_boundary_aligned"] = r_boundary_aligned
        rooms_info[r_i]["order_aligned"] = int(order[r_i])

    # adjacencies_aligned
    adjacencies_aligned = np.array(edges_aligned[:, :2]).tolist()
    data_json_aligned["adjacencies_aligned"] = adjacencies_aligned

    if support_boundary:
        # doors & windows
        doors = np.array(data_mat["doors"]).tolist()
        data_json_aligned["doors"] = doors
        windows = np.array(data_mat["windows"]).tolist()
        data_json_aligned["windows"] = windows
    else:
        boundary_aligned = np.array(data_mat["boundary_aligned"]).tolist()
        data_json_aligned["boundary_aligned"] = boundary_aligned

    return data_json_aligned


def main(data_json, support_boundary=True):

    data_mat = get_datamat_from_json(data_json, support_boundary)
    
    if support_boundary:
        boxes_aligned, order, room_boundaries = align_fp(data_mat['boundary'], data_mat['boxes'], data_mat['types'], data_mat['edges'], threshold = 18)
    else:
        boxes_aligned, order, room_boundaries, boundary_aligned = align_fp_woboundary(data_mat['boxes'], data_mat['types'], data_mat['edges'], threshold = 12)
    
    boxes_aligned_yxyx = (boxes_aligned[:,[1,0,3,2]]).astype(int)
    edges_aligned = np.array(get_adjancenies_from_boxes(boxes_aligned_yxyx), dtype=np.int32)

    room_boundaries = [item if item is not None else [] for item in room_boundaries]

    data_mat['boxes_aligned'] = boxes_aligned
    data_mat['edges_aligned'] = edges_aligned
    data_mat['order'] = order
    data_mat['room_boundaries'] = room_boundaries

    if support_boundary:
        doors, windows = get_dw(data_mat)
        data_mat['doors'] = doors
        data_mat['windows'] = windows
    else:
        data_mat['boundary_aligned'] = boundary_aligned

    data_json_aligned = get_datajson_aligned_from_mat(data_mat, data_json, support_boundary)

    return data_json_aligned


if __name__ == '__main__':

    name_json = 'b'
    support_boundary = True
    # name_json = '_'
    # support_boundary = False

    syn_dataset_path = f'output_json/{name_json}.json'

    with open(syn_dataset_path, encoding='utf-8') as f:
        syn_dataset_json = json.load(f)

    outputs = []
    for data_json in syn_dataset_json:
        name_fp = data_json["name"]
        try:
            outputs.append(main(data_json, support_boundary))
        except Exception as e:
            print(f"error name_fp: {name_fp}")
        if len(outputs) % 1000 == 0:
            print(len(outputs))

    print(len(outputs))
    openw = open(syn_dataset_path, 'w')
    json.dump(outputs, openw, ensure_ascii=False)
    openw.close()