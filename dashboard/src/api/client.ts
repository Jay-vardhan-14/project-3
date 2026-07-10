import axios from 'axios';

const api = axios.create({ baseURL: '/api/v1', timeout: 15000 });

export type DriftStatus = 'healthy' | 'warning' | 'critical';
export type Severity = 'info' | 'warning' | 'critical';
export type RunStatus = 'running' | 'success' | 'failed';

export interface MetricsSummary {
  predictions_today: number;
  avg_latency_ms: number;
  model_version: string | null;
  model_f1: number | null;
  drift_status: DriftStatus;
}

export interface VolumePoint {
  bucket: string;
  count: number;
}

export interface LatencyStats {
  p50: number;
  p95: number;
  p99: number;
  count: number;
}

export interface DriftPoint {
  report_date: string;
  drift_score: number;
  dataset_drift_detected: boolean;
  features_drifted: number;
  total_features: number;
  prediction_drift_detected: boolean;
  report_path: string | null;
}

export interface DistributionSlice {
  predicted_sentiment: string;
  count: number;
}

export interface ConfidenceBucket {
  bucket: number;
  count: number;
}

export interface RecentPrediction {
  created_at: string;
  predicted_sentiment: string;
  confidence: number;
  latency_ms: number;
  model_version: string;
}

export interface AlertRow {
  id: string;
  alert_type: string;
  severity: Severity;
  message: string;
  is_resolved: boolean;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface PipelineRunRow {
  id: string;
  dag_id: string;
  run_id: string;
  status: RunStatus;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  metrics: Record<string, unknown> | null;
}

export interface ExperimentRow {
  run_id: string;
  run_name: string;
  experiment: string;
  status: string;
  start_time: number;
  metrics: Record<string, number>;
}

export interface ModelRow {
  name: string;
  version: string;
  stage: string;
  run_id: string;
  f1_macro: string | null;
  accuracy: string | null;
}

export interface ModelInfo {
  name: string;
  version: string | null;
  stage: string;
  loaded: boolean;
  flavor: string | null;
  f1_score: number | null;
  accuracy: number | null;
  load_time: string | null;
}

const get = async <T>(url: string, params?: Record<string, unknown>): Promise<T> => {
  const response = await api.get<T>(url, { params });
  return response.data;
};

export const client = {
  summary: () => get<MetricsSummary>('/metrics/summary'),
  volume: (days = 7) => get<VolumePoint[]>('/metrics/predictions', { days }),
  latency: (days = 7) => get<LatencyStats>('/metrics/latency', { days }),
  drift: (days = 30) => get<DriftPoint[]>('/metrics/drift', { days }),
  distribution: (days = 30) => get<DistributionSlice[]>('/metrics/distribution', { days }),
  confidence: (days = 30) => get<ConfidenceBucket[]>('/metrics/distribution/confidence', { days }),
  recentPredictions: (limit = 20) => get<RecentPrediction[]>('/metrics/recent-predictions', { limit }),
  alerts: () => get<AlertRow[]>('/metrics/alerts'),
  pipelineRuns: () => get<PipelineRunRow[]>('/metrics/pipeline-runs'),
  experiments: () => get<ExperimentRow[]>('/metrics/experiments'),
  models: () => get<ModelRow[]>('/metrics/models'),
  modelInfo: () => get<ModelInfo>('/model/info'),
};
