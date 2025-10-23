import { apiClientConfig } from '@/config/api';
import { WebTask } from '@/types/api';

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
};

export const fetchDatasets = async (): Promise<string[]> => {
  const response = await fetch(apiClientConfig.buildUrl('/datasets'));
  return handleResponse<string[]>(response);
};

export const fetchTasksInDataset = async (dataset: string): Promise<string[]> => {
  const response = await fetch(
    apiClientConfig.buildUrl(`/datasets/${encodeURIComponent(dataset)}`),
  );
  return handleResponse<string[]>(response);
};

export const fetchTask = async (taskname: string): Promise<WebTask> => {
  const response = await fetch(
    apiClientConfig.buildUrl(`/tasks/${encodeURIComponent(taskname)}`),
  );
  return handleResponse<WebTask>(response);
};
