import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import { WebGrid } from '@/types/api';

interface GridProps {
  grid: WebGrid;
}

const Grid = ({ grid }: GridProps) => {
  return (
    <div className="flex w-full items-start gap-6">
      <div className="flex-1">
        <GridCells grid={grid.cells} />
      </div>
      <div className="min-w-[12rem] shrink-0">
        <GridData gridData={grid.data} />
      </div>
    </div>
  );
};

export default Grid;
