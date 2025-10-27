export type Palette = Record<number, string> & Record<string, string>;

export interface ColoredGrid {
  cells: number[][];
  palette?: Palette;
  width: number;
  height: number;
}

export type MetricPrimitive = number | string | boolean | null;

export interface MetricDictionary {
  [key: string]: MetricValue;
}

export type MetricValue = MetricPrimitive | MetricDictionary;

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
export type HeatmapSets = Record<string, HeatmapCollection>;

export interface WebIOPair {
  input: WebGrid;
  output: WebGrid;
  heatmap_sets: HeatmapSets;
}

export interface WebTask {
  train: WebIOPair[];
  test: WebIOPair[];
}
