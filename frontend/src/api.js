import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const uploadFile = (file) => {
  const fd = new window.FormData();
  fd.append('file', file);
  return api.post('/data/upload', fd);
};

export const runEDA = (data) => api.post('/analysis/eda', data);

export const trainModel = (data) => api.post('/train/train', data);

export const trainAndSaveModel = (data) => api.post('/artifacts/save', data);

export const listAlgorithms = () => api.get('/train/algorithms');

export const autoTune = (data) => api.post('/tuning/auto-tune', data);

export const getLLMStatus = () => api.get('/settings/llm');

export const listRuns = (experimentId) => api.get('/artifacts/runs', { params: experimentId ? { experiment_id: experimentId } : {} });

export const downloadModel = (experimentId, runId) => `/api/artifacts/runs/${experimentId}/${runId}/download`;
