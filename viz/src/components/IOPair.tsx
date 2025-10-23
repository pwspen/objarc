import GridCells from '@/components/GridCells';
import GridData from '@/components/GridData';
import { WebIOPair } from '@/types/api';

interface IOPairProps {
  pair: WebIOPair;
  label?: string;
}

const IOPair = ({ pair, label }: IOPairProps) => {
  return (
    <section className="flex flex-col gap-4 items-center">
      <div className="grid grid-cols-[auto_minmax(12rem,1fr)] grid-rows-[auto_auto_auto] items-start gap-x-6 gap-y-4">
        <div className="row-start-1 flex justify-center">
          <GridCells grid={pair.input.cells} />
        </div>
        <div className="row-start-1 col-start-2">
          <GridData gridData={pair.input.data} />
        </div>
        <div className="row-start-2 flex justify-center text-slate-400">
          <span className="text-4xl leading-none">↓</span>
        </div>
        <div className="row-start-2 col-start-2" aria-hidden="true" />
        <div className="row-start-3 flex justify-center">
          <GridCells grid={pair.output.cells} />
        </div>
        <div className="row-start-3 col-start-2">
          <GridData gridData={pair.output.data} />
        </div>
      </div>
    </section>
  );
};

export default IOPair;
