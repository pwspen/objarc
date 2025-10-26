import { useMemo } from 'react';

import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import HeatmapDisplay from '@/components/HeatmapDisplay';
import { HeatmapGrid, WebIOPair } from '@/types/api';

const HEATMAP_MATCHERS = [
  { id: 'inputAuto', term: 'input auto', label: 'Input Auto' },
  { id: 'cross', term: 'cross', label: 'Cross' },
  { id: 'outputAuto', term: 'output auto', label: 'Output Auto' },
] as const;

type HeatmapMatcher = (typeof HEATMAP_MATCHERS)[number];

interface IOPairProps {
  pair: WebIOPair;
  label?: string;
  selectedHeatmapSet: string | null;
}

interface HeatmapGrids {
  inputAuto: HeatmapGrid;
  cross: HeatmapGrid;
  outputAuto: HeatmapGrid;
}

const IOPair = ({ pair, label, selectedHeatmapSet }: IOPairProps) => {
  const heatmapGrids = useMemo<HeatmapGrids | null>(() => {
    if (!selectedHeatmapSet) {
      return null;
    }

    const heatmapSets = pair.heatmap_sets ?? {};
    const selectedSet = heatmapSets[selectedHeatmapSet];
    if (!selectedSet) {
      const context = label ? ` for IO pair ${label}` : '';
      throw new Error(`Unknown heatmap set "${selectedHeatmapSet}"${context}`);
    }

    const entries = Object.entries(selectedSet);
    if (entries.length > 3) {
      const context = label ? ` for IO pair ${label}` : '';
      throw new Error(`Heatmap set "${selectedHeatmapSet}"${context} contains more than 3 heatmaps`);
    }

    const normalizedEntries = entries.map(([key, value]) => ({
      originalKey: key,
      normalized: key.toLowerCase(),
      value,
    }));
    const usedKeys = new Set<string>();
    const matchedGrids: Record<HeatmapMatcher['id'], HeatmapGrid | undefined> = {
      inputAuto: undefined,
      cross: undefined,
      outputAuto: undefined,
    };

    for (const matcher of HEATMAP_MATCHERS) {
      const match = normalizedEntries.find(
        (entry) => entry.normalized.includes(matcher.term) && !usedKeys.has(entry.originalKey),
      );
      if (!match) {
        const context = label ? ` for IO pair ${label}` : '';
        throw new Error(
          `Missing required "${matcher.label}" heatmap in set "${selectedHeatmapSet}"${context}`,
        );
      }
      usedKeys.add(match.originalKey);
      matchedGrids[matcher.id] = match.value;
    }

    return {
      inputAuto: matchedGrids.inputAuto!,
      cross: matchedGrids.cross!,
      outputAuto: matchedGrids.outputAuto!,
    };
  }, [pair.heatmap_sets, selectedHeatmapSet, label]);

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
