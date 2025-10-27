const normalizeBaseUrl = (value: string) => value.replace(/\/+$/, '');

const deriveDefaultBaseUrl = () => {
  const baseUrl = import.meta.env.BASE_URL ?? '/';
  const trimmedBase = baseUrl.replace(/\/+$/, '');
  if (trimmedBase.length === 0) {
    return '/arc/api';
  }
  return `${trimmedBase}/arc/api`;
};

const resolveBaseUrl = (): string => {
  const envValue = import.meta.env.VITE_API_BASE_URL;
  if (typeof envValue === 'string' && envValue.trim().length > 0) {
    return normalizeBaseUrl(envValue.trim());
  }
  return normalizeBaseUrl(deriveDefaultBaseUrl());
};

const API_BASE_URL = resolveBaseUrl();

const buildUrl = (path: string) => {
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${suffix}`;
};

export const apiClientConfig = {
  baseUrl: API_BASE_URL,
  buildUrl,
};
