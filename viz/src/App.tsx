import { useCallback, useEffect, useMemo, useState } from 'react';
import TaskView from '@/components/TaskView';
import { WebTask } from '@/types/api';
import { fetchDatasets, fetchTask, fetchTasksInDataset } from '@/api/client';
import '@/App.css';

const App = () => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [tasks, setTasks] = useState<string[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [task, setTask] = useState<WebTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingTasks, setLoadingTasks] = useState(false);
  const [loadingTask, setLoadingTask] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoadingDatasets(true);
    setError(null);

    fetchDatasets()
      .then((list) => {
        if (cancelled) return;
        setDatasets(list);
        setSelectedDataset((current) => {
          if (current && list.includes(current)) {
            return current;
          }
          return list.length > 0 ? list[0] : null;
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load datasets');
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingDatasets(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDataset) {
      setTasks([]);
      setSelectedTask(null);
      setTask(null);
      return;
    }

    let cancelled = false;
    setLoadingTasks(true);
    setTasks([]);
    setSelectedTask(null);
    setTask(null);
    setError(null);

    fetchTasksInDataset(selectedDataset)
      .then((list) => {
        if (cancelled) return;
        const sorted = [...new Set(list)].sort((a, b) => a.localeCompare(b)); // deduplicate. might have been api bug that was causing duplication? idk
        setTasks(sorted);
        setSelectedTask((current) => {
          if (current && sorted.includes(current)) {
            return current;
          }
          return sorted[0] ?? null;
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load tasks');
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingTasks(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDataset]);

  useEffect(() => {
    if (!selectedTask) {
      setTask(null);
      return;
    }

    let cancelled = false;
    setLoadingTask(true);
    setError(null);

    fetchTask(selectedTask)
      .then((data) => {
        if (cancelled) return;
        setTask(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load task');
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingTask(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTask]);

  const datasetOptions = useMemo(() => {
    if (loadingDatasets && datasets.length === 0) {
      return [<option key="loading" value="">Loading datasets…</option>];
    }

    if (datasets.length === 0) {
      return [<option key="none" value="">No datasets available</option>];
    }

    return datasets.map((name) => (
      <option key={name} value={name}>
        {name}
      </option>
    ));
  }, [datasets, loadingDatasets]);

  const taskOptions = useMemo(() => {
    if (loadingTasks || tasks.length === 0) {
      const label = loadingTasks ? 'Loading tasks…' : 'No tasks available';
      return [<option key="placeholder" value="">{label}</option>];
    }

    return tasks.map((name) => (
      <option key={name} value={name}>
        {name}
      </option>
    ));
  }, [tasks, loadingTasks]);

  const selectTaskByOffset = useCallback(
    (offset: number) => {
      if (tasks.length === 0) return;

      if (!selectedTask) {
        const fallbackIndex = offset >= 0 ? 0 : tasks.length - 1;
        setSelectedTask(tasks[fallbackIndex]);
        return;
      }

      const currentIndex = tasks.indexOf(selectedTask);
      if (currentIndex === -1) {
        setSelectedTask(tasks[0]);
        return;
      }

      const nextIndex = (currentIndex + offset + tasks.length) % tasks.length;
      setSelectedTask(tasks[nextIndex]);
    },
    [selectedTask, tasks],
  );

  const handleNextTask = useCallback(() => selectTaskByOffset(1), [selectTaskByOffset]);
  const handlePreviousTask = useCallback(
    () => selectTaskByOffset(-1),
    [selectTaskByOffset],
  );

  const handleRandomTask = () => {
    if (tasks.length === 0) return;
    const pool =
      tasks.length > 1 && selectedTask
        ? tasks.filter((name) => name !== selectedTask)
        : tasks;
    const next = pool[Math.floor(Math.random() * pool.length)];
    setSelectedTask(next);
  };

  const taskNavigationDisabled = tasks.length === 0 || loadingTasks;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-8">
        <header className="flex flex-wrap items-end gap-4 rounded-lg border border-slate-800 bg-slate-900/80 p-4">
          <div className="flex flex-col gap-1">
            <label htmlFor="dataset-select" className="text-xs uppercase tracking-wide text-slate-400">
              Dataset
            </label>
            <select
              id="dataset-select"
              value={selectedDataset ?? ''}
              onChange={(event) => setSelectedDataset(event.target.value || null)}
              disabled={loadingDatasets || datasets.length === 0}
              className="rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
            >
              {datasetOptions}
            </select>
          </div>

          <div className="ml-auto flex items-end gap-2">
            <div className="flex flex-col gap-1">
              <label htmlFor="task-select" className="text-xs uppercase tracking-wide text-slate-400">
                Task
              </label>
              <select
                id="task-select"
                value={selectedTask ?? ''}
                onChange={(event) => setSelectedTask(event.target.value || null)}
                disabled={loadingTasks || tasks.length === 0}
                className="min-w-[14rem] rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
              >
                {taskOptions}
              </select>
            </div>
            <button
              type="button"
              onClick={handlePreviousTask}
              disabled={taskNavigationDisabled}
              className="rounded border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-700 disabled:opacity-50"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={handleNextTask}
              disabled={taskNavigationDisabled}
              className="rounded border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-700 disabled:opacity-50"
            >
              Next
            </button>
            <button
              type="button"
              onClick={handleRandomTask}
              disabled={taskNavigationDisabled}
              className="rounded border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-700 disabled:opacity-50"
            >
              Random
            </button>
          </div>
        </header>

        {error ? (
          <div className="rounded border border-red-600 bg-red-900/30 p-4 text-sm text-red-200">
            {error}
          </div>
        ) : null}

        <main className="flex-1 overflow-auto rounded-lg border border-slate-800 bg-slate-900/60 p-6">
          {loadingTask ? (
            <div className="flex h-full items-center justify-center text-slate-400">
              Loading task…
            </div>
          ) : (
            <TaskView task={task} />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
