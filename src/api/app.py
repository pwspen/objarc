from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

import numpy as np

from api.schemas import (
    ColoredGrid,
    HeatmapGrid,
    WebGrid,
    WebGridData,
    WebIOPair,
    WebTask,
)
from api.services import get_valid_datasets, load_task_names

from backend import (
    ArcIOPair,
    ArcTask,
    auto_correlation,
    cross_correlation,
    get_grid_stats,
    print_matrix,
    EMPTY_COLOR,
    entropy_filter,
)


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


def heatmaps_arr(image_a: np.ndarray, image_b: np.ndarray) -> list[np.ndarray]:
    inp_auto = auto_correlation(image_a, center=True)
    cross = cross_correlation(image_a, image_b, center=True)
    out_auto = auto_correlation(image_b, center=True)

    return [inp_auto, cross, out_auto]


def heatmaps(
    image_a: np.ndarray, image_b: np.ndarray, remove_most_common: bool = False
) -> dict[str, HeatmapGrid]:
    if remove_most_common:
        image_a, image_b = image_a.copy(), image_b.copy()
        for img in [image_a, image_b]:
            unique, counts = np.unique(img, return_counts=True)
            most_common = unique[np.argmax(counts[unique != EMPTY_COLOR])]
            img[img == most_common] = EMPTY_COLOR

    names = ["Input Auto", "Cross", "Output Auto"]
    return {
        name: HeatmapGrid(values=hm.tolist())
        for name, hm in zip(names, heatmaps_arr(image_a, image_b))
    }


filters = {
    "2x2": np.array([[1, 1], [1, 1]]),
    "2x1": np.array([[1, 1]]),
    "1x2": np.array([[1], [1]]),
    "3x3": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
}


def ent_heatmaps(
    image_a: np.ndarray, image_b: np.ndarray, filters: dict[str, np.ndarray]
) -> dict[str, dict[str, HeatmapGrid]]:
    labels = ["Input", "Output"]
    EMPTY_GRID = HeatmapGrid(values=[])
    results = {}
    ents = []
    for name, filt in filters.items():
        ent_key = f"Entropy: {name}"
        results[ent_key] = {}
        filters_valid = True
        for img, label in [(image_a, labels[0]), (image_b, labels[1])]:
            imshape = img.shape
            fishape = filt.shape
            if imshape[0] < fishape[0] or imshape[1] < fishape[1]:
                filters_valid = False
                results[ent_key][label] = EMPTY_GRID
            else:
                ent = entropy_filter(img, filt)
                ents.append(ent)
                results[ent_key][label] = HeatmapGrid(values=ent.tolist())

        ent_corr_key = f"Correlation: {name} entropy"
        if filters_valid:
            results[ent_corr_key] = heatmaps(ents[0], ents[1], remove_most_common=False)
        else:
            results[ent_corr_key] = {
                "Input Auto": EMPTY_GRID,
                "Cross": EMPTY_GRID,
                "Output Auto": EMPTY_GRID,
            }

    return results


def _to_web_task(task: ArcTask) -> WebTask:
    def to_web_io_pair(pair: ArcIOPair) -> WebIOPair:
        inp, out = pair.to_lists()

        fft_inp = pair.input.copy() + 1
        fft_out = pair.output.copy() + 1

        all_heatmaps = ent_heatmaps(fft_inp, fft_out, filters)
        all_heatmaps.update(
            {
                "Correlation: naive": heatmaps(
                    fft_inp, fft_out, remove_most_common=False
                ),
                "Correlation: Most common removed": heatmaps(
                    fft_inp, fft_out, remove_most_common=True
                ),
            }
        )

        return WebIOPair(
            input=WebGrid(
                cells=ColoredGrid(cells=inp),
                data=WebGridData(data=get_grid_stats(pair.input)),
            ),
            output=WebGrid(
                cells=ColoredGrid(cells=out),
                data=WebGridData(data=get_grid_stats(pair.output)),
            ),
            heatmap_sets=all_heatmaps,
        )

    web_train = [to_web_io_pair(pair) for pair in task.train_pairs]
    web_test = [to_web_io_pair(pair) for pair in task.test_pairs]
    return WebTask(train=web_train, test=web_test)


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8010,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
