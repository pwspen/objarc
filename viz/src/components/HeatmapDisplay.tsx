import { useMemo } from 'react';

import type { HeatmapGrid } from '@/types/api';

const DEFAULT_MAX_DIMENSION = 180; // px
const DEFAULT_LEGEND_MIN_HEIGHT = 96; // px

interface HeatmapDisplayProps {
  grid: HeatmapGrid;
  label: string;
  maxDimension?: number;
  legendMinHeight?: number;
}

interface HeatmapSummary {
  rows: number;
  cols: number;
  min: number;
  max: number;
  haveData: boolean;
  longestSide: number;
}

interface ColorStop {
  stop: number;
  hex: string;
  rgb: [number, number, number];
}

const VIRIDIS_COLOR_STOPS: ColorStop[] = [
  { stop: 0.0, hex: '#440154', rgb: [68, 1, 84] },
  { stop: 0.1, hex: '#482878', rgb: [72, 40, 120] },
  { stop: 0.2, hex: '#3e4a89', rgb: [62, 74, 137] },
  { stop: 0.3, hex: '#31688e', rgb: [49, 104, 142] },
  { stop: 0.4, hex: '#26828e', rgb: [38, 130, 142] },
  { stop: 0.5, hex: '#1f9e89', rgb: [31, 158, 137] },
  { stop: 0.6, hex: '#35b779', rgb: [53, 183, 121] },
  { stop: 0.7, hex: '#6ece58', rgb: [110, 206, 88] },
  { stop: 0.8, hex: '#b5de2b', rgb: [181, 222, 43] },
  { stop: 0.9, hex: '#fde724', rgb: [253, 231, 36] },
  { stop: 1.0, hex: '#fde724', rgb: [253, 231, 36] },
];

const COLOR_STOPS = VIRIDIS_COLOR_STOPS;

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const interpolateColor = (value: number): string => {
  const stops = COLOR_STOPS;
  if (value <= 0) return stops[0].hex;
  if (value >= 1) return stops[stops.length - 1].hex;

  const upperIndex = stops.findIndex((stop) => stop.stop >= value);
  if (upperIndex <= 0) {
    return stops[upperIndex === -1 ? stops.length - 1 : upperIndex].hex;
  }

  const lower = stops[upperIndex - 1];
  const upper = stops[upperIndex];
  const localRange = upper.stop - lower.stop || 1;
  const t = clamp((value - lower.stop) / localRange, 0, 1);

  const mixChannel = (index: number) =>
    Math.round(lower.rgb[index] + (upper.rgb[index] - lower.rgb[index]) * t);

  const r = mixChannel(0);
  const g = mixChannel(1);
  const b = mixChannel(2);

  const toHex = (channel: number) => channel.toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

const summarizeValues = (values: number[][]): HeatmapSummary => {
  const rows = values.length;
  const cols = rows > 0 ? values[0]?.length ?? 0 : 0;
  if (rows === 0 || cols === 0) {
    return {
      rows: 0,
      cols: 0,
      min: 0,
      max: 0,
      haveData: false,
      longestSide: 0,
    };
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const row of values) {
    for (const value of row) {
      if (Number.isFinite(value)) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 0;
  }

  return {
    rows,
    cols,
    min,
    max,
    haveData: true,
    longestSide: Math.max(rows, cols),
  };
};

const formatLegendValue = (value: number) => {
  if (!Number.isFinite(value)) {
    return '–';
  }
  return value === 0 ? '0' : value.toPrecision(3);
};

const buildLegendGradient = () =>
  `linear-gradient(to top, ${COLOR_STOPS.map((stop) => `${stop.hex} ${stop.stop * 100}%`).join(', ')})`;

const HeatmapDisplay = ({
  grid,
  label,
  maxDimension = DEFAULT_MAX_DIMENSION,
  legendMinHeight = DEFAULT_LEGEND_MIN_HEIGHT,
}: HeatmapDisplayProps) => {
  const values = grid.values ?? [];
  const summary = useMemo(() => summarizeValues(values), [values]);

  const cellSize = useMemo(() => {
    if (!summary.haveData || summary.longestSide === 0) {
      return 0;
    }
    return maxDimension / summary.longestSide;
  }, [summary, maxDimension]);

  const legendGradient = useMemo(() => buildLegendGradient(), []);

  const midpoint = summary.haveData ? (summary.min + summary.max) / 2 : 0;

  return (
    <div className="space-y-2">
      <div className="text-sm font-semibold text-slate-100">{label}</div>
      {!summary.haveData || cellSize === 0 ? (
        <div className="text-xs italic text-slate-500">no heatmap data</div>
      ) : (
        <div className="flex flex-row">
          <div
            className="inline-grid gap-px border border-slate-800/70 bg-slate-900/60 p-1"
            style={{
              gridTemplateColumns: `repeat(${summary.cols}, ${cellSize}px)`,
              gridAutoRows: `${cellSize}px`,
            }}
          >
            {values.map((row, rowIndex) =>
              row.map((value, colIndex) => {
                const normalized =
                  summary.max === summary.min
                    ? 0.5
                    : clamp((value - summary.min) / (summary.max - summary.min), 0, 1);
                const color = interpolateColor(normalized);
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className="border border-slate-800"
                    style={{ backgroundColor: color }}
                    aria-label={`value ${value}`}
                  />
                );
              }),
            )}
          </div>
          <div className="flex items-stretch gap-3">
            <div
              className="w-4 shrink-0 rounded"
              style={{ background: legendGradient, minHeight: legendMinHeight }}
              aria-hidden="true"
            />
            <div className="flex flex-col justify-between text-xs text-slate-300">
              <span>{formatLegendValue(summary.max)}</span>
              <span>{formatLegendValue(midpoint)}</span>
              <span>{formatLegendValue(summary.min)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapDisplay;
