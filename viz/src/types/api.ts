export type Palette = Record<string, string>;

export interface ColoredGrid {
  cells: number[][];
  palette?: Palette;
}

export type MetricValue = number | MetricDictionary;

export interface MetricDictionary {
  [key: string]: MetricValue;
}

export interface WebGridData {
  data: MetricDictionary;
}

export interface WebGrid {
  cells: ColoredGrid;
  data: WebGridData;
}

export interface HeatmapGrid {
  values: number[][];
}

export type HeatmapCollection = Record<string, HeatmapGrid>;

export interface WebIOPair {
  input: WebGrid;
  output: WebGrid;
  heatmaps: HeatmapCollection;
}

export interface WebTask {
  train: WebIOPair[];
  test: WebIOPair[];
}
