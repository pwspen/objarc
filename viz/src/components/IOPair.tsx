import { useMemo } from 'react';

import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import HeatmapDisplay from '@/components/HeatmapDisplay';
import { HeatmapGrid, WebIOPair } from '@/types/api';

const REQUIRED_HEATMAP_KEYS = ['Input Auto', 'Cross', 'Output Auto'] as const;

type RequiredHeatmapKey = (typeof REQUIRED_HEATMAP_KEYS)[number];

interface IOPairProps {
  pair: WebIOPair;
  label?: string;
  showHeatmaps: boolean;
}

interface HeatmapGrids {
  inputAuto: HeatmapGrid;
  cross: HeatmapGrid;
  outputAuto: HeatmapGrid;
}

const IOPair = ({ pair, label, showHeatmaps }: IOPairProps) => {
  const heatmapGrids = useMemo<HeatmapGrids | null>(() => {
    if (!showHeatmaps) {
      return null;
    }

    const missing = REQUIRED_HEATMAP_KEYS.filter(
      (key) => pair.heatmaps[key as RequiredHeatmapKey] === undefined,
    );

    if (missing.length > 0) {
      const context = label ? ` for IO pair ${label}` : '';
      throw new Error(`Missing required heatmaps [${missing.join(', ')}]${context}`);
    }

    return {
      inputAuto: pair.heatmaps['Input Auto']!,
      cross: pair.heatmaps['Cross']!,
      outputAuto: pair.heatmaps['Output Auto']!,
    };
  }, [pair.heatmaps, showHeatmaps, label]);

  const gridTemplateClass = heatmapGrids
    ? 'grid-cols-[auto_max-content_minmax(11rem,1fr)]'
    : 'grid-cols-[auto_max-content]';

  return (
    <section className="flex flex-col gap-4 items-center">
      {label ? (
        <header>
          <h2 className="text-lg font-medium text-slate-200">{label}</h2>
        </header>
      ) : null}
      <div
        className={`grid ${gridTemplateClass} grid-rows-[auto_auto_auto] items-start gap-x-6 gap-y-4`}
      >
        <div className="row-start-1 flex justify-center">
          <GridCells grid={pair.input.cells} />
        </div>
        <div className="row-start-1 col-start-2">
          <GridData gridData={pair.input.data} />
        </div>
        {heatmapGrids ? (
          <div className="row-start-1 col-start-3">
            <HeatmapDisplay label="Input Auto" grid={heatmapGrids.inputAuto} />
          </div>
        ) : null}
        <span className="row-start-2 col-start-1 place-self-center text-6xl leading-none text-slate-400">
          ↓
        </span>
        <div className="row-start-2 col-start-2" aria-hidden="true" />
        {heatmapGrids ? (
          <div className="row-start-2 col-start-3">
            <HeatmapDisplay label="Cross" grid={heatmapGrids.cross} />
          </div>
        ) : null}
        <div className="row-start-3 flex justify-center">
          <GridCells grid={pair.output.cells} />
        </div>
        <div className="row-start-3 col-start-2">
          <GridData gridData={pair.output.data} />
        </div>
        {heatmapGrids ? (
          <div className="row-start-3 col-start-3">
            <HeatmapDisplay label="Output Auto" grid={heatmapGrids.outputAuto} />
          </div>
        ) : null}
      </div>
    </section>
  );
};

export default IOPair;
