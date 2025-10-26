from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

import numpy as np

from .schemas import ColoredGrid, HeatmapGrid, WebGrid, WebGridData, WebIOPair, WebTask
from .services import get_valid_datasets, load_task_names

from src.backend import ArcIOPair, ArcTask, auto_correlation, cross_correlation, get_grid_stats, print_matrix, EMPTY_COLOR


def create_app() -> FastAPI:
    app = FastAPI()
    api_router = APIRouter(prefix="/api")

    @api_router.get("/datasets", response_model=list[str])
    def datasets() -> list[str]:
        return get_valid_datasets()

    @api_router.get("/datasets/{dataset_name}", response_model=list[str])
    def tasks_in_dataset(dataset_name: str) -> list[str]:
        return load_task_names(dataset_name)

    @api_router.get("/tasks/{taskname}", response_model=WebTask)
    def get_task(taskname: str) -> WebTask:
        task = ArcTask.from_name(taskname)
        return _to_web_task(task)

    app.include_router(api_router, prefix="/arc")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "https://synapsomorphy.com"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

def heatmaps(image_a: np.ndarray, image_b: np.ndarray, remove_most_common: bool = False) -> dict[str, HeatmapGrid]:
    if remove_most_common:
        image_a, image_b = image_a.copy(), image_b.copy()
        for img in [image_a, image_b]:
            unique, counts = np.unique(img, return_counts=True)
            most_common = unique[np.argmax(counts[unique != EMPTY_COLOR])]
            img[img == most_common] = EMPTY_COLOR

    inp_auto = auto_correlation(image_a, center=True)
    cross = cross_correlation(image_a, image_b, center=True)
    out_auto = auto_correlation(image_b, center=True)

    return {
        "Input Auto": HeatmapGrid(values=inp_auto.tolist()), 
        "Cross": HeatmapGrid(values=cross.tolist()),
        "Output Auto": HeatmapGrid(values=out_auto.tolist()),
        }

def _to_web_task(task: ArcTask) -> WebTask:
    def to_web_io_pair(pair: ArcIOPair) -> WebIOPair:
        inp, out = pair.to_lists()

        fft_inp = pair.input.copy() + 1
        fft_out = pair.output.copy() + 1

        inp_auto = auto_correlation(fft_inp)
        cross = cross_correlation(fft_inp, fft_out)
        out_auto = auto_correlation(fft_out)

        return WebIOPair(
            input=WebGrid(cells=ColoredGrid(cells=inp), data=WebGridData(data=get_grid_stats(pair.input))),
            output=WebGrid(cells=ColoredGrid(cells=out), data=WebGridData(data=get_grid_stats(pair.output))),
            heatmap_sets=
            {
                "Naive": heatmaps(fft_inp, fft_out, remove_most_common=False),
                "Most common removed": heatmaps(fft_inp, fft_out, remove_most_common=True),
            }
        )

    web_train = [to_web_io_pair(pair) for pair in task.train_pairs]
    web_test = [to_web_io_pair(pair) for pair in task.test_pairs]
    return WebTask(train=web_train, test=web_test)


app = create_app()

uvicorn.run(app, host="127.0.0.1", port=8010)
