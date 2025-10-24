import { useMemo } from 'react';

import { ColoredGrid } from '@/types/api';

import ColorGrid from './ColorGrid';

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

const GridCells = ({ grid }: GridCellsProps) => {
  const palette = useMemo(
    () => ({ ...defaultPalette, ...grid.palette }),
    [grid.palette],
  );

  const colors = useMemo(
    () =>
      (grid.cells ?? []).map((row) =>
        row.map((value) => palette[value.toString()] ?? defaultPalette['0']),
      ),
    [grid.cells, palette],
  );

  return <ColorGrid colors={colors} fallbackColor={defaultPalette['0']} />;
};

export default GridCells;
