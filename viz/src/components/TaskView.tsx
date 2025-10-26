import Pairs from '@/components/Pairs';
import { WebTask } from '@/types/api';

interface TaskViewProps {
  task: WebTask | null;
  selectedHeatmapSet: string | null;
}

const TaskView = ({ task, selectedHeatmapSet }: TaskViewProps) => {
  if (!task) {
    return (
      <div className="flex h-full items-center justify-center text-slate-400">
        Select a task to view its examples.
      </div>
    );
  }

  return (
    <div className="space-y-12">
      <Pairs pairs={task.train} name="Train" selectedHeatmapSet={selectedHeatmapSet} />
      <div className="border-t border-slate-700" />
      <Pairs pairs={task.test} name="Test" selectedHeatmapSet={selectedHeatmapSet} />
    </div>
  );
};

export default TaskView;
