import IOPair from '@/components/IOPair';
import { WebIOPair } from '@/types/api';

interface PairProps {
  pairs: WebIOPair[];
  name: string;
}

const Pairs = ({ pairs, name }: PairProps) => {
  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold text-slate-100">{name}</h2>
      <div className="space-y-10">
        {pairs.map((pair, index) => (
          <IOPair key={index} pair={pair} label={`${name} Pair ${index + 1}`} />
        ))}
      </div>
    </section>
  );
};

export default Pairs;
