const DEFAULT_API_BASE_URL = 'http://localhost:8000';

const normalizeBaseUrl = (value: string) => value.replace(/\/+$/, '');

const resolveBaseUrl = (): string => {
  const envValue = import.meta.env.VITE_API_BASE_URL;
  if (typeof envValue === 'string' && envValue.trim().length > 0) {
    return normalizeBaseUrl(envValue.trim());
  }
  return DEFAULT_API_BASE_URL;
};

const API_BASE_URL = normalizeBaseUrl(resolveBaseUrl());

const buildUrl = (path: string) => {
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${suffix}`;
};

export const apiClientConfig = {
  baseUrl: API_BASE_URL,
  buildUrl,
};

