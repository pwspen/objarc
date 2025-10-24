from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.backend import ArcDataset, ArcTask, ArcIOPair, Grid, get_grid_stats, fft_cross_correlation
from src.api.services import load_task_names, get_valid_datasets
import numpy as np

arc_colors = {
    0: '#000000',  # Black
    1: '#0000FF',  # Blue
    2: '#FF0000',  # Red
    3: '#00FF00',  # Green
    4: '#FFFF00',  # Yellow
    5: '#808080',  # Gray
    6: '#FFC0CB',  # Pink
    7: '#FFA500',  # Orange
    8: '#00FFFF',  # Cyan
    9: '#A52A2A',  # Brown
}

class ColoredGrid(BaseModel):
    cells: list[list[int]] = Field(..., description="Cell values in row-major order")
    palette: dict[int, str] = Field(default=arc_colors, description="Mapping of cell values to hex colors")
    
    @model_validator(mode='before')
    @classmethod
    def extract_dimensions(cls, data):
        if isinstance(data, dict) and 'cells' in data and 'width' not in data:
            cells_2d = data['cells']
            
            # Compute dimensions
            data['height'] = len(cells_2d)
            data['width'] = len(cells_2d[0]) if cells_2d else 0
        
        return data
    
    @model_validator(mode='after')
    def validate_all(self):
        cells = self.cells
        palette = self.palette
        if len(palette) > 10:
            raise ValueError("Palette can have at most 10 colors")
        for key, color in palette.items():
            if not isinstance(key, int) or key < 0 or key > 9:
                raise ValueError("Palette keys must be integers 0-9")
            if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
                raise ValueError(f"Invalid hex color: {color}")
        
        
        max_dim = 30
        if len(cells) > max_dim or any(len(row) > max_dim for row in cells):
            raise ValueError(f"Number of rows exceeds maximum of {max_dim}")

        for val in [cell for row in cells for cell in row]:
            if val not in palette.keys():
                raise ValueError(f"Cell value {val} not in palette (keys: {list(palette.keys())})")
        
        return self
    
class WebGridData(BaseModel):
    data: dict

class WebGrid(BaseModel):
    cells: ColoredGrid
    data: WebGridData

class HeatmapGrid(BaseModel):
    values: list[list[float]]

class WebIOPair(BaseModel):
    input: WebGrid
    output: WebGrid
    heatmaps: dict[str, HeatmapGrid]

class WebTask(BaseModel):
    train: list[WebIOPair]
    test: list[WebIOPair]

app = FastAPI()

api = APIRouter(prefix="/api")

@api.get("/datasets", response_model=list[str])
def get_datasets():
    return get_valid_datasets()

@api.get("/datasets/{dataset_name}", response_model=list[str])
def get_tasks_in_dataset(dataset_name: str):
    return load_task_names(dataset_name)

@api.get("/tasks/{taskname}", response_model=WebTask)
def get_task(taskname: str):
    task = ArcTask.from_name(taskname)
    return to_web_task(task)

app.include_router(api, prefix="/arc")  # final paths: /arc/api/...

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://synapsomorphy.com"],  # add prod origin
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_web_task(task: ArcTask) -> WebTask:
    def to_web_io_pair(pair: ArcIOPair) -> WebIOPair:
        inp, out = pair.to_lists()

        fft_inp = pair.input.copy() + 1
        fft_out = pair.output.copy() + 1

        inp_auto = fft_cross_correlation(fft_inp, fft_inp, center=True)["matches"]
        cross = fft_cross_correlation(fft_inp, fft_out, center=True)["matches"]
        out_auto = fft_cross_correlation(fft_out, fft_out, center=True)["matches"]

        return WebIOPair(
            input=WebGrid(cells=ColoredGrid(cells=inp), data=WebGridData(data=get_grid_stats(pair.input))),
            output=WebGrid(cells=ColoredGrid(cells=out), data=WebGridData(data=get_grid_stats(pair.output))),
            heatmaps={
                "Input Auto": HeatmapGrid(values=inp_auto.tolist()),
                "Cross": HeatmapGrid(values=cross.tolist()),
                "Output Auto": HeatmapGrid(values=out_auto.tolist()),
            },
        )
    
    web_train = [to_web_io_pair(pair) for pair in task.train_pairs]
    web_test = [to_web_io_pair(pair) for pair in task.test_pairs]
    return WebTask(train=web_train, test=web_test)
