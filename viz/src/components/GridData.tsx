import { MetricDictionary, MetricValue, WebGridData } from '@/types/api';

interface DepthStyle {
  lineIndentClass: string;
  keyClass: string;
  valueClass: string;
  groupLabelClass: string;
  nestedContainerClass: string;
}

const depthStyles: DepthStyle[] = [
  {
    lineIndentClass: '',
    keyClass: 'text-sm font-medium text-slate-200',
    valueClass: 'text-sm text-slate-300',
    groupLabelClass: 'text-sm font-semibold text-slate-100',
    nestedContainerClass: 'mt-1 space-y-1 border-l border-slate-800/60 pl-4',
  },
  {
    lineIndentClass: 'pl-4',
    keyClass: 'text-xs font-medium text-slate-200',
    valueClass: 'text-xs text-slate-300',
    groupLabelClass: 'text-xs font-semibold text-slate-200',
    nestedContainerClass: 'mt-1 space-y-1 border-l border-slate-800/50 pl-3',
  },
  {
    lineIndentClass: 'pl-6',
    keyClass: 'text-[0.7rem] font-medium text-slate-200',
    valueClass: 'text-[0.7rem] text-slate-400',
    groupLabelClass: 'text-[0.7rem] font-semibold text-slate-300',
    nestedContainerClass: 'mt-1 space-y-1 border-l border-slate-800/40 pl-3',
  },
];

const getDepthStyle = (depth: number): DepthStyle =>
  depthStyles[Math.min(depth, depthStyles.length - 1)];

const isMetricDictionary = (value: MetricValue): value is MetricDictionary =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

interface GridDataProps {
  gridData: WebGridData;
}

const formatValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  return value.toPrecision(3);
};

const renderEntries = (dictionary: MetricDictionary, depth = 0): JSX.Element[] => {
  const entries = Object.entries(dictionary);
  const style = getDepthStyle(depth);

  return entries.map(([key, value]) => {
    if (typeof value === 'number') {
      return (
        <div
          key={`${depth}-${key}`}
          className={`flex justify-left gap-2 ${style.lineIndentClass}`}
        >
          <span className={style.keyClass}>{key}</span>
          <span className={style.valueClass}>{formatValue(value)}</span>
        </div>
      );
    }

    if (isMetricDictionary(value)) {
      return (
        <div
          key={`${depth}-${key}`}
          className={`space-y-1 ${style.lineIndentClass}`}
        >
          <div className={style.groupLabelClass}>{key}</div>
          <div className={style.nestedContainerClass}>
            {renderEntries(value, depth + 1)}
          </div>
        </div>
      );
    }

    return (
      <div
        key={`${depth}-${key}`}
        className={`flex justify-between gap-2 ${style.lineIndentClass}`}
      >
        <span className={style.keyClass}>{key}</span>
        <span className={style.valueClass}>{String(value)}</span>
      </div>
    );
  });
};

const GridData = ({ gridData }: GridDataProps) => {
  if (Object.keys(gridData.data).length === 0) {
    return <span className="text-sm text-slate-400">no metrics</span>;
  }

  return <div className="space-y-2">{renderEntries(gridData.data)}</div>;
};

export default GridData;
