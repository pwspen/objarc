from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi import FastAPI

from classes import ArcDataset, ArcTask, ArcIOPair, Grid
from utils import load_tasknames, valid_datasets

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
    
    @field_validator('palette')
    @classmethod
    def validate_palette(cls, v):
        if len(v) > 10:
            raise ValueError("Palette can have at most 10 colors")
        for key, color in v.items():
            if not isinstance(key, int) or key < 0 or key > 9:
                raise ValueError("Palette keys must be integers 0-9")
            if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
                raise ValueError(f"Invalid hex color: {color}")
        return v
    
    @field_validator('cells')
    @classmethod
    def validate_cells(cls, v, info):
        palette = info.data.get('palette', {})
        
        max_dim = 30
        if len(v) > max_dim or any(len(row) > max_dim for row in v):
            raise ValueError(f"Number of rows exceeds maximum of {max_dim}")

        for val in [cell for row in v for cell in row]:
            if val not in palette:
                raise ValueError(f"Cell value {val} not in palette")
        return v
    
class WebGridData(BaseModel):
    data: dict[str, float]

class WebGrid(BaseModel):
    cells: ColoredGrid
    data: WebGridData

class WebIOPair(BaseModel):
    input: WebGrid
    output: WebGrid

class WebTask(BaseModel):
    train: list[WebIOPair]
    test: list[WebIOPair]

app = FastAPI()

@app.get("/datasets", response_model=list[str])
def get_datasets():
    return valid_datasets

@app.get("/datasets/{dataset_name}", response_model=list[str])
def get_tasks_in_dataset(dataset_name: str):
    return load_tasknames(dataset_name)

@app.get("/tasks/{taskname}", response_model=WebTask)
def get_task(taskname: str):
    task = ArcTask.from_name(taskname)

    return to_web_task(task)

def to_web_task(task: ArcTask) -> WebTask:
    def to_web_io_pair(pair: ArcIOPair) -> WebIOPair:
        inp, out = pair.to_lists()
        return WebIOPair(
            input=WebGrid(cells=ColoredGrid(cells=inp), data=WebGridData(data={})),
            output=WebGrid(cells=ColoredGrid(cells=out), data=WebGridData(data={}))
        )
    
    web_train = [to_web_io_pair(pair) for pair in task.train_pairs]
    web_test = [to_web_io_pair(pair) for pair in task.test_pairs]
    return WebTask(train=web_train, test=web_test)