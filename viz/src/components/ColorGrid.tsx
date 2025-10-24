import { useMemo } from 'react';

const DEFAULT_MAX_DIMENSION = 180; // px
const DEFAULT_LEGEND_MIN_HEIGHT = 96; // px

interface LegendLabels {
  max: string;
  mid: string;
  min: string;
}

export interface ColorGridLegend {
  gradient: string;
  labels: LegendLabels;
  minHeight?: number;
}

export interface ColorGridProps {
  colors: string[][];
  maxDimension?: number;
  legend?: ColorGridLegend;
  className?: string;
  fallbackColor?: string;
}

interface GridSummary {
  rows: number;
  cols: number;
  longestSide: number;
  haveData: boolean;
}

const summarizeGrid = (colors: string[][]): GridSummary => {
  const rows = colors.length;
  const cols = rows > 0 ? colors[0]?.length ?? 0 : 0;
  if (rows === 0 || cols === 0) {
    return {
      rows: 0,
      cols: 0,
      longestSide: 1,
      haveData: false,
    };
  }
  return {
    rows,
    cols,
    longestSide: Math.max(rows, cols),
    haveData: true,
  };
};

const ColorGrid = ({
  colors,
  maxDimension = DEFAULT_MAX_DIMENSION,
  legend,
  className,
  fallbackColor = '#000000',
}: ColorGridProps) => {
  const summary = useMemo(() => summarizeGrid(colors), [colors]);
  const cellSize = useMemo(() => {
    if (!summary.haveData || summary.longestSide === 0) {
      return 24;
    }
    return maxDimension / summary.longestSide;
  }, [summary, maxDimension]);

  const containerClass = ['flex flex-row', className].filter(Boolean).join(' ');

  const templateColumns = summary.haveData ? summary.cols : 1;

  return (
    <div className={containerClass}>
      <div
        className="inline-grid gap-0.5 border border-slate-700 bg-slate-900/60 p-1"
        style={{
          gridTemplateColumns: `repeat(${templateColumns}, ${cellSize}px)`,
          gridAutoRows: `${cellSize}px`,
        }}
      >
        {summary.haveData
          ? colors.map((row, rowIndex) =>
              row.map((color, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="border border-slate-800"
                  style={{ backgroundColor: color }}
                />
              )),
            )
          : (
            <div
              className="border border-slate-800"
              style={{
                backgroundColor: fallbackColor,
                width: `${cellSize}px`,
                height: `${cellSize}px`,
              }}
            />
          )}
      </div>
      {legend ? (
        <div className="ml-3 flex items-stretch gap-3">
          <div
            className="w-4 shrink-0 rounded"
            style={{
              background: legend.gradient,
              minHeight: legend.minHeight ?? DEFAULT_LEGEND_MIN_HEIGHT,
            }}
            aria-hidden="true"
          />
          <div className="flex flex-col justify-between text-xs text-slate-300">
            <span>{legend.labels.max}</span>
            <span>{legend.labels.mid}</span>
            <span>{legend.labels.min}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default ColorGrid;
