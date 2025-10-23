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

export interface WebIOPair {
  input: WebGrid;
  output: WebGrid;
}

export interface WebTask {
  train: WebIOPair[];
  test: WebIOPair[];
}
