import { ColoredGrid } from '@/types/api';
import { useMemo } from 'react';

interface GridCellsProps {
  grid: ColoredGrid;
}

const defaultPalette: Record<string, string> = {
  '0': '#000000',
  '1': '#0000FF',
  '2': '#FF0000',
  '3': '#00FF00',
  '4': '#FFFF00',
  '5': '#808080',
  '6': '#FFC0CB',
  '7': '#FFA500',
  '8': '#00FFFF',
  '9': '#A52A2A',
};

const CELL_SIZE_REM = 1.5;
const CELL_SIZE = `${CELL_SIZE_REM}rem`;

const GridCells = ({ grid }: GridCellsProps) => {
  const rows = grid.cells;
  const displayRows = useMemo(
    () =>
      rows.length === 0
        ? rows
        : rows
            .slice()
            .reverse()
            .map((row) => row.slice().reverse()),
    [rows],
  );
  const palette = useMemo(
    () => ({ ...defaultPalette, ...grid.palette }),
    [grid.palette],
  );
  const columnCount = displayRows[0]?.length ?? 0;
  const effectiveColumns = columnCount > 0 ? columnCount : 1;
  const resolveColor = (value: number) =>
    palette[value.toString()] ?? defaultPalette['0'];
  const emptyCellStyle = { width: CELL_SIZE, height: CELL_SIZE };

  return (
    <div
      className="inline-grid gap-0.5 border border-slate-700 bg-slate-900/60 p-1"
      style={{
        gridTemplateColumns: `repeat(${effectiveColumns}, ${CELL_SIZE})`,
        gridAutoRows: CELL_SIZE,
      }}
    >
      {rows.length === 0 || columnCount === 0 ? (
        <div
          className="border border-slate-800"
          style={{ ...emptyCellStyle, backgroundColor: defaultPalette['0'] }}
        />
      ) : null}
      {displayRows.map((row, rowIndex) =>
        row.map((value, colIndex) => {
          const key = `${rowIndex}-${colIndex}`;
          return (
            <div
              key={key}
              className="border border-slate-800"
              style={{ backgroundColor: resolveColor(value) }}
            />
          );
        }),
      )}
    </div>
  );
};

export default GridCells;
