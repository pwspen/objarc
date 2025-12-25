import { useEffect, useMemo, useState } from 'react';

import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import { WebGrid } from '@/types/api';

interface GridProps {
  grid: WebGrid;
}

const Grid = ({ grid }: GridProps) => {
  const [appliedCount, setAppliedCount] = useState(0);
  const rects = grid.rects ?? [];

  useEffect(() => {
    setAppliedCount(0);
  }, [grid]);

  const displayedGrid = useMemo(() => {
    const baseCells = grid.cells?.cells ?? [];
    const cloned = baseCells.map((row) => [...row]);
    const target = Math.min(appliedCount, rects.length);

    for (let i = 0; i < target; i += 1) {
      const { r1, c1, r2, c2 } = rects[i];
      for (let r = r1; r <= r2 && r < cloned.length; r += 1) {
        if (r < 0) continue;
        const row = cloned[r];
        for (let c = c1; c <= c2 && c < row.length; c += 1) {
          if (c < 0) continue;
          row[c] = -1;
        }
      }
    }

    return { ...grid.cells, cells: cloned };
  }, [appliedCount, grid.cells, rects]);

  const canStepBack = appliedCount > 0;
  const canStepForward = appliedCount < rects.length;

  return (
    <div className="flex w-full items-start gap-6">
      <div className="flex flex-1 flex-col items-center gap-3">
        <GridCells grid={displayedGrid} />
        <div className="flex gap-2">
          <button
            type="button"
            className="rounded text-lg border border-slate-700 px-3 py-1 text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => setAppliedCount((count) => Math.max(0, count - 1))}
            disabled={!canStepBack}
            aria-label="Apply previous rectangle"
          >
            ←
          </button>
          <button
            type="button"
            className="rounded text-lg border border-slate-700 px-3 py-1 text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() =>
              setAppliedCount((count) => Math.min(rects.length, count + 1))
            }
            disabled={!canStepForward}
            aria-label="Apply next rectangle"
          >
            →
          </button>
        </div>
      </div>
      <div className="min-w-[12rem] shrink-0">
        <GridData gridData={grid.data} />
      </div>
    </div>
  );
};

export default Grid;
