"""Run the repository's NodeDiff -> AdjacencyDiff -> PartitioningDiff pipeline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("node_diff", "adjacency_diff", "partitioning_diff", "output"):
    sys.path.insert(0, str(ROOT / folder))

from node_diff.script_util import model_and_diffusion_defaults as node_defaults, create_model_and_diffusion as make_node
from adjacency_diff.script_util import model_and_diffusion_defaults as adjacency_defaults, create_model_and_diffusion as make_adjacency
from partitioning_diff.script_util import model_and_diffusion_defaults as partition_defaults, create_model_and_diffusion as make_partition

CANVAS = 256
MAX_ROOMS = 8


def _expand_polygon(points: list[list[int]], size: int = 40) -> list[list[int]]:
    points = [list(map(int, p)) for p in points]
    if points[0] != points[-1]:
        points.append(points[0])
    if len(points) > size:
        raise ValueError(f"Boundary has too many vertices ({len(points)-1}); at most {size-1} are supported.")
    while len(points) < size:
        i = max(range(len(points)-1), key=lambda j: (points[j][0]-points[j+1][0])**2 + (points[j][1]-points[j+1][1])**2)
        a, b = points[i], points[i+1]
        points.insert(i+1, [round((a[0]+b[0])/2), round((a[1]+b[1])/2)])
    return points


def _user_record(boundary: list[list[int]], entrance: list[list[int]]) -> dict:
    if len(boundary) < 3:
        raise ValueError("Boundary must contain at least three [x, y] points.")
    if len(entrance) != 2:
        raise ValueError("Entrance must be two [x, y] endpoints on one boundary edge.")
    pts = np.asarray(boundary, dtype=int)
    if np.any(pts <= 0) or np.any(pts >= CANVAS):
        raise ValueError("Boundary coordinates must be strictly between 0 and 256, matching the training data.")
    if not (np.all(pts[:, 0] == pts[0, 0]) or np.all(pts[:, 1] == pts[0, 1])):
        # The repository supports polygon boundaries; only the entrance is axis aligned.
        pass
    ent = np.asarray(entrance, dtype=int)
    if not (np.all(ent > 0) and np.all(ent < CANVAS)):
        raise ValueError("Entrance coordinates must be strictly between 0 and 256.")
    if ent[0, 0] == ent[1, 0]:
        x = int(ent[0, 0]); y0, y1 = sorted(map(int, ent[:, 1]))
        entrance_expand = [[x-8,y0],[x+8,y0],[x+8,y1],[x-8,y1]]
    elif ent[0, 1] == ent[1, 1]:
        y = int(ent[0, 1]); x0, x1 = sorted(map(int, ent[:, 0]))
        entrance_expand = [[x0,y-8],[x1,y-8],[x1,y+8],[x0,y+8]]
    else:
        raise ValueError("Entrance endpoints must form a horizontal or vertical segment.")
    return {"name":"user_floorplan", "boundary":pts.tolist(), "boundary_expand":_expand_polygon(pts.tolist()), "entrance_expand":entrance_expand}


def _conditions(record: dict, device: torch.device) -> dict:
    b = (np.asarray(record['boundary_expand']).flatten() / CANVAS * 2 - 1).astype(np.float32)
    d = (np.asarray(record['entrance_expand']).flatten() / CANVAS * 2 - 1).astype(np.float32)
    return {
        "cond_boundary": torch.tensor(np.tile(b, (MAX_ROOMS, 1))[None], device=device),
        "cond_door": torch.tensor(np.tile(d, (MAX_ROOMS, 1))[None], device=device),
        "cond_number": torch.zeros((1, MAX_ROOMS, MAX_ROOMS), device=device),
    }


def _load_model(factory, defaults, checkpoint: str, stage: str):
    config = defaults()
    config.update(dataset="rplan", support_boundary=True, support_partial=False)
    # Training scripts call update_arg_parser(), which sets this to 512.
    config["num_channels"] = 512
    if stage == "node": config["support_conditions"] = ""
    elif stage == "adjacency": config["support_conditions"] = "ncsl"
    else: config["support_conditions"] = "ncsla"
    config["set_name"] = "test"
    if stage == "node": config.update(input_channels=5, out_channels=5)
    elif stage == "adjacency": config.update(input_channels=8, out_channels=8)
    else: config.update(input_channels=4, out_channels=4)
    model, diffusion = factory(**config)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, diffusion, device


def _sample(diffusion, model, channels: int, kwargs: dict, device: torch.device) -> torch.Tensor:
    # Sampling calls the model with is_syn=True; all model conditions must use
    # the syn_ names used by the repository's test Dataset and sample scripts.
    kwargs = {f"syn_{key}": value for key, value in kwargs.items()}
    kwargs.setdefault("syn_atten_mask", torch.zeros((1, MAX_ROOMS, MAX_ROOMS), device=device))
    kwargs.setdefault("syn_padding_mask", torch.zeros((1, MAX_ROOMS), device=device))
    x = torch.randn((1, channels, MAX_ROOMS), device=device)
    with torch.inference_mode():
        samples = diffusion.p_sample_loop(model, x.shape, noise=x, clip_denoised=True, model_kwargs=kwargs)
    # This repository's p_sample_loop returns the final denoising frames as
    # [frames, batch, channels, rooms]; the original sample scripts select [-1].
    if samples.ndim == 4:
        samples = samples[-1]
    if samples.ndim != 3:
        raise RuntimeError(f"Expected a [batch, channels, rooms] sample, got shape {tuple(samples.shape)}")
    return samples


def _decode_nodes(tensor: torch.Tensor, record: dict) -> dict:
    a = tensor[0].permute(1, 0).detach().cpu().numpy()
    rooms = []
    for row in a:
        if row[0] < .5: continue
        cat = int(np.clip(round(((row[1]+1)/2)*6)-1, 0, 5))
        rooms.append({"id":len(rooms), "category":cat, "size":int(round(((row[2]+1)/2)*CANVAS*CANVAS)), "location":np.round(((row[3:5]+1)/2)*CANVAS).astype(int).tolist()})
    if not rooms: raise ValueError("NodeDiff returned no rooms; cannot continue the pipeline.")
    return {**record, "rooms":rooms}


def _room_conditions(data: dict, device: torch.device):
    rooms = data["rooms"]; n=len(rooms); area=CANVAS*CANVAS
    number=np.zeros((MAX_ROOMS,MAX_ROOMS),np.float32)
    node=np.full((MAX_ROOMS,4),-1.,np.float32)
    for i,r in enumerate(rooms):
        number[i,i]=1
        node[i]=[(r['category']+1)/3-1, r['size']/area*2-1, r['location'][0]/CANVAS*2-1, r['location'][1]/CANVAS*2-1]
    mask=np.ones((MAX_ROOMS,MAX_ROOMS),np.float32); mask[:n,:n]=0
    return {"cond_number":torch.tensor(number[None],device=device),"cond_node":torch.tensor(node[None],device=device),"atten_mask":torch.tensor(mask[None],device=device),"padding_mask":torch.tensor(np.r_[np.zeros(n),np.ones(MAX_ROOMS-n)][None],device=device)}


def _adjacency_conditions(data: dict, device: torch.device):
    c=_room_conditions(data,device); n=len(data['rooms'])
    c["cond_adjacency"]=torch.full((1,MAX_ROOMS,MAX_ROOMS),-1.,device=device)
    return c


def _decode_adjacencies(tensor: torch.Tensor, data: dict) -> dict:
    matrix=tensor[0].detach().cpu().numpy(); n=len(data['rooms']); edges=[]
    for i in range(n):
        for j in range(i+1,n):
            if matrix[i,j] > 0: edges.append([i,j])
    return {**data,"adjacencies":edges}


class FloorPlanInference:
    def __init__(self, checkpoints: dict[str,str] | None = None):
        self.checkpoints = checkpoints or {
            "node":os.getenv("DIFFPLANNER_NODE_CHECKPOINT", str(ROOT/"node_diff/scripts/trained_model/b_model300000.pt")),
            "adjacency":os.getenv("DIFFPLANNER_ADJACENCY_CHECKPOINT", str(ROOT/"adjacency_diff/scripts/trained_model/bncsl_model300000.pt")),
            "partition":os.getenv("DIFFPLANNER_PARTITION_CHECKPOINT", str(ROOT/"partitioning_diff/scripts/trained_model/bncsla_model300000.pt")),
        }
        self.models = None

    def _initialize(self):
        if self.models is not None: return
        for stage, path in self.checkpoints.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing {stage} checkpoint: {path}. Set DIFFPLANNER_{stage.upper()}_CHECKPOINT to its location.")
        self.models = (
            _load_model(make_node,node_defaults,self.checkpoints['node'],'node'),
            _load_model(make_adjacency,adjacency_defaults,self.checkpoints['adjacency'],'adjacency'),
            _load_model(make_partition,partition_defaults,self.checkpoints['partition'],'partition'),
        )

    def generate(self, boundary: list[list[int]], entrance: list[list[int]], num_results: int = 1) -> list[dict]:
        if not 1 <= int(num_results) <= 5:
            raise ValueError("Number of results must be between 1 and 5.")
        return [self._generate_one(boundary, entrance) for _ in range(int(num_results))]

    def _generate_one(self, boundary: list[list[int]], entrance: list[list[int]]) -> dict:
        record=_user_record(boundary,entrance)
        self._initialize()
        from output.post_processing import main as align_plan
        from output.visualization import draw_bubble, vis_floorplan
        from PIL import Image
        import io
        nm,ad,pm=self.models
        nmodel,ndiff,device=nm; amodel,adiff,_=ad; pmodel,pdiff,_=pm
        common=_conditions(record,device)
        node=_sample(ndiff,nmodel,5,{**common,"cond_category":torch.zeros((1,MAX_ROOMS,6),device=device),"cond_partial":torch.full((1,MAX_ROOMS,5),-1.,device=device)},device)
        data1=_decode_nodes(node,record)
        nc=_room_conditions(data1,device)
        adj=_sample(adiff,amodel,8,{**common,**nc,"cond_partial":torch.full((1,MAX_ROOMS,MAX_ROOMS),-1.,device=device)},device)
        data2=_decode_adjacencies(adj,data1)
        pc=_room_conditions(data2,device); pc["cond_adjacency"]=torch.full((1,MAX_ROOMS,MAX_ROOMS),-1.,device=device)
        for i,j in data2['adjacencies']: pc['cond_adjacency'][0,i,j]=1; pc['cond_adjacency'][0,j,i]=1
        boxes=_sample(pdiff,pmodel,4,{**common,**pc,"cond_partial":torch.full((1,MAX_ROOMS,4),-1.,device=device)},device)[0].permute(1,0).cpu().numpy()
        for i,room in enumerate(data2['rooms']): room['box']=np.round(((boxes[i]+1)/2)*CANVAS).astype(int).tolist()
        data3=align_plan(data2, support_boundary=True)
        def png(arr):
            im=Image.fromarray(np.uint8(arr)); b=io.BytesIO(); im.save(b,format='PNG'); return b.getvalue()
        boundary_arr=np.asarray(record['boundary'],dtype=np.int32)
        def bubble(rooms, edges):
            import cv2
            image=draw_bubble([],rooms,edges,canvas_size=512)
            polygon=(boundary_arr[:,:2]*2).astype(np.int32)
            cv2.polylines(image,[polygon.reshape(-1,1,2)],True,(60,60,60),2)
            door=(np.asarray(entrance,dtype=np.int32)*2).astype(np.int32)
            cv2.line(image,tuple(door[0]),tuple(door[1]),(40,90,220),5)
            return png(image)
        stage1=bubble(data1['rooms'],[])
        stage2=bubble(data2['rooms'],data2['adjacencies'])
        stage3=png(vis_floorplan(data3,is_syn=True,support_boundary=True,canvas_size=512))
        return {"stage1":stage1,"stage2":stage2,"stage3":stage3,"data":data3}
