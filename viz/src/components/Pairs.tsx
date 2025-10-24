import IOPair from '@/components/IOPair';
import { WebIOPair } from '@/types/api';

interface PairProps {
  pairs: WebIOPair[];
  name: string;
  showHeatmaps: boolean;
}

const Pairs = ({ pairs, name, showHeatmaps }: PairProps) => {
  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold text-slate-100">{name}</h2>
      <div className="space-y-10">
        {pairs.map((pair, index) => (
          <div key={index} className="space-y-6">
            <IOPair
              pair={pair}
              label={`${name} Pair ${index + 1}`}
              showHeatmaps={showHeatmaps}
            />
            {index < pairs.length - 1 ? (
              <div className="mx-auto w-3/4 border-t border-slate-700/60" />
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
};

export default Pairs;
