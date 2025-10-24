import { useMemo } from 'react';

import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import HeatmapDisplay from '@/components/HeatmapDisplay';
import { WebIOPair } from '@/types/api';

const REQUIRED_HEATMAP_KEYS = ['Input Auto', 'Cross', 'Output Auto'] as const;

type RequiredHeatmapKey = (typeof REQUIRED_HEATMAP_KEYS)[number];

interface IOPairProps {
  pair: WebIOPair;
  label?: string;
  showHeatmaps: boolean;
}

const IOPair = ({ pair, label, showHeatmaps }: IOPairProps) => {
  const heatmaps = useMemo(() => {
    if (!showHeatmaps) {
      return null;
    }

    const missing = REQUIRED_HEATMAP_KEYS.filter(
      (key) => pair.heatmaps[key as RequiredHeatmapKey] === undefined,
    );

    if (missing.length > 0) {
      throw new Error(
        `Missing required heatmaps [${missing.join(', ')}] for IO pair ${label ?? ''}`.trim(),
      );
    }

    return REQUIRED_HEATMAP_KEYS.map((key) => ({
      key,
      grid: pair.heatmaps[key as RequiredHeatmapKey],
    }));
  }, [pair.heatmaps, showHeatmaps, label]);

  const gridTemplateClass = showHeatmaps
    ? 'grid-cols-[auto_minmax(12rem,1fr)_minmax(11rem,1fr)]'
    : 'grid-cols-[auto_minmax(12rem,1fr)]';

  return (
    <section className="flex flex-col gap-4 items-center">
      <div
        className={`grid ${gridTemplateClass} grid-rows-[auto_auto_auto] items-start gap-x-6 gap-y-4`}
      >
        <div className="row-start-1 flex justify-center">
          <GridCells grid={pair.input.cells} />
        </div>
        <div className="row-start-1 col-start-2">
          <GridData gridData={pair.input.data} />
        </div>
        {showHeatmaps && heatmaps ? (
          <div className="row-start-1 col-start-3">
            <HeatmapDisplay label="Input Auto" grid={heatmaps[0].grid} />
          </div>
        ) : null}
        <div className="row-start-2 flex justify-center text-slate-400">
          <span className="text-4xl leading-none">↓</span>
        </div>
        <div className="row-start-2 col-start-2" aria-hidden="true" />
        {showHeatmaps && heatmaps ? (
          <div className="row-start-2 col-start-3">
            <HeatmapDisplay label="Cross" grid={heatmaps[1].grid} />
          </div>
        ) : null}
        <div className="row-start-3 flex justify-center">
          <GridCells grid={pair.output.cells} />
        </div>
        <div className="row-start-3 col-start-2">
          <GridData gridData={pair.output.data} />
        </div>
        {showHeatmaps && heatmaps ? (
          <div className="row-start-3 col-start-3">
            <HeatmapDisplay label="Output Auto" grid={heatmaps[2].grid} />
          </div>
        ) : null}
      </div>
    </section>
  );
};

export default IOPair;
