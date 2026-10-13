import { useState, useEffect } from 'react';
import { ArrowRight, ArrowLeft, Play, Gauge, Database, AlertCircle, Download } from 'lucide-react';
import { trainAndSaveModel, listAlgorithms, downloadModel } from '../api';
import { ScatterChart, Scatter, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';

function ScatterPlot({ title, points, metrics, color }) {
  if (!points || points.length === 0) return null;
  const allVals = points.flatMap((p) => [p.y_true, p.y_pred]);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;
  const domain = [minV - range * 0.05, maxV + range * 0.05];

  return (
    <div className="card p-4">
      <h4 className="text-sm font-semibold mb-1">{title}</h4>
      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
        R²={metrics.r2?.toFixed(4) ?? 'N/A'} &nbsp; RMSE={metrics.rmse?.toFixed(4) ?? 'N/A'} &nbsp; MAE={metrics.mae?.toFixed(4) ?? 'N/A'} &nbsp; n={metrics.n ?? points.length}
      </p>
      <div style={{ aspectRatio: '1', maxHeight: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
            <XAxis
              type="number"
              dataKey="y_true"
              name="真实值"
              domain={domain}
              tick={{ fontSize: 10, fill: '#566477' }}
              label={{ value: '真实值', position: 'bottom', offset: 0, fontSize: 11, fill: '#566477' }}
            />
            <YAxis
              type="number"
              dataKey="y_pred"
              name="预测值"
              domain={domain}
              tick={{ fontSize: 10, fill: '#566477' }}
              label={{ value: '预测值', angle: -90, position: 'left', offset: 0, fontSize: 11, fill: '#566477' }}
            />
            <ReferenceLine y={domain[0]} x={domain[0]} segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]} />
            <ReferenceLine
              segment={[
                { x: domain[0], y: domain[0] },
                { x: domain[1], y: domain[1] },
              ]}
              stroke="#888"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
            <Tooltip
              contentStyle={{ background: '#fff', border: '1px solid #DDE3EB', borderRadius: 6, fontSize: 12 }}
              formatter={(val, name) => [val?.toFixed(4), name]}
            />
            <Scatter data={points} fill={color} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function TrainStep({ data, selection, trainResult, setTrainResult, onNext, onPrev, error, setError }) {
  const [algos, setAlgos] = useState({});
  const [algorithm, setAlgorithm] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    listAlgorithms().then((res) => setAlgos(res.data)).catch(() => {});
  }, []);

  const taskAlgos = algos[selection?.taskType] || {};

  const handleTrain = async () => {
    if (!algorithm) return setError('请选择算法');
    setError(null);
    setLoading(true);
    try {
      const res = await trainAndSaveModel({
        filename: data.filename,
        features: selection.features,
        target: selection.target,
        task_type: selection.taskType,
        algorithm,
      });
      setTrainResult(res.data);
    } catch (e) {
      setError('训练失败：' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const formatMetric = (key, val) => {
    if (typeof val !== 'number') return val;
    if (key === 'r2') return val.toFixed(4);
    if (key === 'rmse' || key === 'mae') return val.toFixed(4);
    if (val <= 1) return (val * 100).toFixed(1) + '%';
    return val.toFixed(4);
  };

  const isRegression = selection?.taskType === 'regression';
  const trainPoints = trainResult?.train_predictions || [];
  const testPoints = trainResult?.test_predictions || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">训练模型</h2>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          选择算法进行训练，查看真实值-预测值散点图
        </p>
      </div>

      {error && (
        <div className="error-banner flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Algorithm selector */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Database size={14} style={{ color: 'var(--primary)' }} />
            <h3 className="font-semibold text-sm">算法选择 ({selection?.taskType})</h3>
          </div>
          <div className="max-h-72 overflow-y-auto space-y-0.5">
            {Object.keys(taskAlgos).map((name) => (
              <button
                key={name}
                onClick={() => setAlgorithm(name)}
                className="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors"
                style={{
                  background: algorithm === name ? 'rgba(37,99,235,0.08)' : 'transparent',
                  color: algorithm === name ? 'var(--primary)' : 'var(--text)',
                  fontWeight: algorithm === name ? 600 : 400,
                  border: algorithm === name ? '1px solid rgba(37,99,235,0.2)' : '1px solid transparent',
                }}
              >
                {name}
              </button>
            ))}
          </div>
          <button
            onClick={handleTrain}
            disabled={loading || !algorithm}
            className="btn-primary w-full justify-center mt-3"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'white', borderTopColor: 'transparent' }} />
            ) : (
              <>
                <Play size={14} /> 开始训练
              </>
            )}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {!trainResult ? (
            <div className="card empty-state py-12">
              <Gauge size={36} className="mx-auto mb-2" />
              <p className="text-sm">训练模型后查看结果</p>
            </div>
          ) : (
            <>
              {/* Metrics row */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="badge badge-primary">{trainResult.algorithm}</span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    划分: {trainResult.split_ratio} (训练 {trainResult.train_size} / 测试 {trainResult.test_size})
                  </span>
                </div>
                {trainResult.run_id && (
                  <a
                    href={downloadModel(trainResult.experiment_id, trainResult.run_id)}
                    className="btn-secondary text-xs"
                    download
                  >
                    <Download size={12} /> 下载模型
                  </a>
                )}
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(trainResult.metrics).map(([key, val]) => (
                  <div key={key} className="metric-card">
                    <div className="metric-label">{key.toUpperCase()}</div>
                    <div className="metric-value">{formatMetric(key, val)}</div>
                  </div>
                ))}
              </div>

              {/* Scatter plots for regression */}
              {isRegression && (trainPoints.length > 0 || testPoints.length > 0) && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {trainPoints.length > 0 && (
                    <ScatterPlot
                      title="训练集"
                      points={trainPoints}
                      metrics={trainResult.train_metrics || {}}
                      color="#0072B2"
                    />
                  )}
                  {testPoints.length > 0 && (
                    <ScatterPlot
                      title="测试集"
                      points={testPoints}
                      metrics={trainResult.test_metrics || {}}
                      color="#D55E00"
                    />
                  )}
                </div>
              )}

              {/* Params */}
              <div className="card p-4">
                <h4 className="text-sm font-semibold mb-2">生效参数</h4>
                <pre className="text-xs overflow-x-auto p-3 rounded-lg" style={{ background: 'var(--surface-alt)', color: 'var(--primary)' }}>
                  {JSON.stringify(trainResult.params, null, 2)}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="flex justify-between pt-2">
        <button onClick={onPrev} className="btn-secondary">
          <ArrowLeft size={14} /> 上一步
        </button>
        <button onClick={onNext} className="btn-primary">
          LLM 调参 <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
