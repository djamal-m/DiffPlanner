"""FastAPI service with an embedded Gradio interface."""
from __future__ import annotations

import json
import traceback
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference.pipeline import FloorPlanInference, MAX_ROOMS

engine = FloorPlanInference()
app = FastAPI(title="DiffPlanner", description="Three-stage vector floor-plan generation")


class GenerateRequest(BaseModel):
    boundary: list[list[int]]
    entrance: list[list[int]]
    num_results: int = 1
    node_mode: str = "b"
    room_count: int | None = None
    room_categories: list[int] | None = None


@app.get("/health")
def health():
    return {"status": "ready", "checkpoints_loaded": bool(engine.models)}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        results = engine.generate(request.boundary, request.entrance, request.num_results,
                                  request.node_mode, request.room_count, request.room_categories)
        return [{"images": {k: f"data:image/png;base64,{__import__('base64').b64encode(result[k]).decode()}" for k in ("stage1", "stage2", "stage3")}, "floorplan": result["data"]} for result in results]
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400 if isinstance(exc, ValueError) else 503, detail=str(exc)) from exc


def _gui_generate(boundary_text: str, entrance_text: str, node_mode: str,
                  room_count: float, categories_text: str, num_results: int):
    try:
        boundary=json.loads(boundary_text); entrance=json.loads(entrance_text)
        categories = json.loads(categories_text) if categories_text.strip() else None
        results=engine.generate(boundary,entrance,num_results,node_mode,
                                int(room_count) if room_count is not None else None,categories)
        from PIL import Image
        import io
        def image(blob): return Image.open(io.BytesIO(blob)).copy()
        return [image(r["stage1"]) for r in results],[image(r["stage2"]) for r in results],[image(r["stage3"]) for r in results],json.dumps([r["data"] for r in results],indent=2),""
    except Exception as exc:
        stack = traceback.format_exc()
        print(stack, flush=True)
        return [], [], [], "", stack


with gr.Blocks(title="DiffPlanner") as demo:
    gr.Markdown("# Floor Plan Generator\nEnter a polygon and entrance using pixel coordinates on the 256 × 256 canvas. NodeDiff modes: `b` generates all room attributes, `n` takes a room count, and `nc` takes a room count plus category IDs (0 living room, 1 bedroom, 2 kitchen, 3 bathroom, 4 balcony, 5 storage).")
    boundary=gr.Textbox(label="Boundary vertices (JSON list of [x, y])", lines=4, value="[[24, 24], [232, 24], [232, 232], [24, 232]]")
    entrance=gr.Textbox(label="Entrance endpoints (JSON list of two [x, y] points on a boundary edge)", value="[[100, 24], [132, 24]]")
    node_mode=gr.Radio(choices=[("Boundary only (b)", "b"), ("Room count (bn)", "n"), ("Room count + categories (bnc)", "nc")], value="b", label="NodeDiff model")
    room_count=gr.Slider(1,MAX_ROOMS,value=4,step=1,label="Room count",visible=False)
    room_categories=gr.Textbox(label="Room category IDs (JSON list in room order)",placeholder="[0, 1, 2, 3]",visible=False)
    def update_node_inputs(mode):
        return gr.update(visible=mode in ("n", "nc")), gr.update(visible=mode == "nc")
    node_mode.change(update_node_inputs,node_mode,[room_count,room_categories])
    num_results=gr.Slider(1,5,value=1,step=1,label="Number of results")
    button=gr.Button("Generate", variant="primary")
    with gr.Row():
        stage1=gr.Gallery(label="Stage 1 · Room bubbles")
        stage2=gr.Gallery(label="Stage 2 · Adjacencies")
        stage3=gr.Gallery(label="Stage 3 · Aligned floor plans")
    details=gr.Code(label="Generated floor-plan data", language="json")
    error_details=gr.Code(label="Error details (copy this traceback)", language="python")
    button.click(_gui_generate,[boundary,entrance,node_mode,room_count,room_categories,num_results],[stage1,stage2,stage3,details,error_details])

app = gr.mount_gradio_app(app, demo, path="/")
