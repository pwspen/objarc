import { useMemo } from 'react';

import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import Grid from '@/components/Grid';
import HeatmapDisplay from '@/components/HeatmapDisplay';
import { HeatmapGrid, WebIOPair } from '@/types/api';

const HEATMAP_MATCHERS = [
  { id: 'inputAuto', term: 'input', label: 'Input Auto' },
  { id: 'cross', term: 'cross', label: 'Cross' },
  { id: 'outputAuto', term: 'output', label: 'Output Auto' },
] as const;

interface IOPairProps {
  pair: WebIOPair;
  label?: string;
  selectedHeatmapSet: string | null;
}

interface HeatmapGrids {
  inputAuto?: HeatmapGrid;
  cross?: HeatmapGrid;
  outputAuto?: HeatmapGrid;
}

const IOPair = ({ pair, label, selectedHeatmapSet }: IOPairProps) => {
  const heatmapGrids = useMemo<HeatmapGrids | null>(() => {
    if (!selectedHeatmapSet) {
      return null;
    }

    const heatmapSets = pair.heatmap_sets ?? {};
    const selectedSet = heatmapSets[selectedHeatmapSet];
    if (!selectedSet) {
      return null;
    }

    const normalizedEntries = Object.entries(selectedSet).map(([key, value]) => ({
      originalKey: key,
      normalized: key.toLowerCase(),
      value,
    }));
    const usedKeys = new Set<string>();
    const matchedGrids: HeatmapGrids = {};

    for (const matcher of HEATMAP_MATCHERS) {
      const match = normalizedEntries.find(
        (entry) => entry.normalized.includes(matcher.term) && !usedKeys.has(entry.originalKey),
      );
      if (!match) {
        continue;
      }
      usedKeys.add(match.originalKey);
      matchedGrids[matcher.id] = match.value;
    }

    const hasMatches = Object.values(matchedGrids).some(Boolean);
    return hasMatches ? matchedGrids : null;
  }, [pair.heatmap_sets, selectedHeatmapSet]);

  const hasHeatmaps =
    heatmapGrids !== null &&
    Object.values(heatmapGrids).some((grid): grid is HeatmapGrid => Boolean(grid));

  const gridTemplateClass = hasHeatmaps
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
        <Grid grid={pair.input} />
        {hasHeatmaps && heatmapGrids?.inputAuto ? (
          <div className="row-start-1 col-start-3">
            <HeatmapDisplay label="Input Auto" grid={heatmapGrids.inputAuto} />
          </div>
        ) : null}
        <span className="row-start-2 col-start-1 place-self-center text-6xl leading-none text-slate-400">
          ↓
        </span>
        <div className="row-start-2 col-start-2" aria-hidden="true" />
        {hasHeatmaps && heatmapGrids?.cross ? (
          <div className="row-start-2 col-start-3">
            <HeatmapDisplay label="Cross" grid={heatmapGrids.cross} />
          </div>
        ) : null}
        <div className="row-start-3 flex justify-center">
          <Grid grid={pair.output} />
        </div>
        {hasHeatmaps && heatmapGrids?.outputAuto ? (
          <div className="row-start-3 col-start-3">
            <HeatmapDisplay label="Output Auto" grid={heatmapGrids.outputAuto} />
          </div>
        ) : null}
      </div>
    </section>
  );
};

export default IOPair;
