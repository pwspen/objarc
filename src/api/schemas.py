from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

DEFAULT_ARC_COLORS: Dict[int, str] = {
    0: "#000000",  # Black
    1: "#0000FF",  # Blue
    2: "#FF0000",  # Red
    3: "#00FF00",  # Green
    4: "#FFFF00",  # Yellow
    5: "#808080",  # Gray
    6: "#FFC0CB",  # Pink
    7: "#FFA500",  # Orange
    8: "#00FFFF",  # Cyan
    9: "#A52A2A",  # Brown
}


class ColoredGrid(BaseModel):
    cells: List[List[int]] = Field(..., description="Cell values in row-major order")
    palette: Dict[int, str] = Field(default_factory=lambda: DEFAULT_ARC_COLORS.copy(), description="Mapping of cell values to hex colors")
    width: int | None = None
    height: int | None = None

    @model_validator(mode="before")
    @classmethod
    def add_dimensions(cls, data):
        if isinstance(data, dict) and "cells" in data:
            cells_2d = data["cells"]
            data.setdefault("height", len(cells_2d))
            data.setdefault("width", len(cells_2d[0]) if cells_2d else 0)
        return data

    @model_validator(mode="after")
    def validate_all(self):
        cells = self.cells
        palette = self.palette
        if len(palette) > 10:
            raise ValueError("Palette can have at most 10 colors")
        for key, color in palette.items():
            if not isinstance(key, int) or key < 0 or key > 9:
                raise ValueError("Palette keys must be integers 0-9")
            if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
                raise ValueError(f"Invalid hex color: {color}")

        max_dim = 30
        if len(cells) > max_dim or any(len(row) > max_dim for row in cells):
            raise ValueError(f"Number of rows exceeds maximum of {max_dim}")

        for val in (cell for row in cells for cell in row):
            if val not in palette.keys():
                raise ValueError(f"Cell value {val} not in palette (keys: {list(palette.keys())})")
        return self


class WebGridData(BaseModel):
    data: dict


class WebGrid(BaseModel):
    cells: ColoredGrid
    data: WebGridData


class HeatmapGrid(BaseModel):
    values: List[List[float]]


class WebIOPair(BaseModel):
    input: WebGrid
    output: WebGrid
    heatmap_sets: Dict[str, Dict[str, HeatmapGrid]]


class WebTask(BaseModel):
    train: List[WebIOPair]
    test: List[WebIOPair]
