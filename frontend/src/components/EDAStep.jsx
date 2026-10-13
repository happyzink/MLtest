import { useState, useEffect, useCallback } from 'react';
import { ArrowLeft, ArrowRight, BarChart3, Flame, Zap, AlertCircle } from 'lucide-react';
import { runEDA } from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLORS = ['#2563EB', '#3B82F6', '#60A5FA', '#0072B2', '#D55E00', '#E5750F', '#16A34A', '#D97706', '#DC2626', '#7C3AED'];

function HeatMap({ matrix, title }) {
  if (!matrix || Object.keys(matrix).length === 0) return null;
  const cols = Object.keys(matrix);

  return (
    <div>
      {title && <h4 className="text-sm font-semibold mb-2">{title}</h4>}
      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-px"
          style={{ gridTemplateColumns: `80px repeat(${cols.length}, minmax(36px, 1fr))` }}
        >
          <div />
          {cols.map((c) => (
            <div key={c} className="text-[10px] text-center truncate px-0.5" style={{ color: 'var(--text-muted)' }}>
              {c}
            </div>
          ))}
          {cols.map((r) => (
            <>
              <div key={`l-${r}`} className="text-[10px] truncate pr-1 flex items-center justify-end" style={{ color: 'var(--text-muted)' }}>
                {r}
              </div>
              {cols.map((c) => {
                const v = matrix[r]?.[c] ?? 0;
                const abs = Math.abs(v);
                const hue = v >= 0 ? 220 : 0;
                const bg = `hsla(${hue}, 70%, 50%, ${abs * 0.6})`;
                return (
                  <div
                    key={`${r}-${c}`}
                    className="text-[10px] text-center py-1 rounded-sm"
                    style={{ background: bg, color: abs > 0.4 ? 'white' : 'var(--text-muted)' }}
                    title={`${r} x ${c}: ${typeof v === 'number' ? v.toFixed(3) : v}`}
                  >
                    {typeof v === 'number' ? v.toFixed(2) : v}
                  </div>
                );
              })}
            </>
          ))}
        </div>
      </div>
    </div>
  );
}

function ImportanceChart({ data, title }) {
  if (!data) return null;
  const chartData = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([name, value]) => ({ name, value: Math.round(value * 10000) / 10000 }));

  return (
    <div>
      {title && <h4 className="text-sm font-semibold mb-2">{title}</h4>}
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 26)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
          <XAxis type="number" tick={{ fontSize: 11, fill: '#566477' }} />
          <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 11, fill: '#566477' }} />
          <Tooltip
            contentStyle={{ background: '#fff', border: '1px solid #DDE3EB', borderRadius: 6, fontSize: 12 }}
          />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function EDAStep({ data, selection, edaResult, setEdaResult, onNext, onPrev, error, setError }) {
  const [loading, setLoading] = useState(false);

  const runAnalysis = useCallback(async () => {
    if (edaResult || !data || !selection) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runEDA({
        filename: data.filename,
        features: selection.features,
        target: selection.target,
        task_type: selection.taskType,
      });
      setEdaResult(res.data);
    } catch (e) {
      setError('分析失败：' + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  }, [data, selection, edaResult, setEdaResult, setError]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    runAnalysis();
  }, [runAnalysis]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">探索性数据分析</h2>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          相关性矩阵、特征重要性与互信息
        </p>
      </div>

      {error && (
        <div className="error-banner flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {loading && (
        <div className="empty-state py-16">
          <div
            className="w-10 h-10 mx-auto border-3 rounded-full animate-spin mb-3"
            style={{ borderColor: 'var(--primary)', borderTopColor: 'transparent', borderWidth: 3 }}
          />
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>正在分析数据...</p>
        </div>
      )}

      {edaResult && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <Flame size={14} style={{ color: 'var(--error)' }} />
                <h3 className="font-semibold text-sm">Pearson 相关系数</h3>
              </div>
              <HeatMap matrix={edaResult.pearson_corr} />
            </div>

            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <Flame size={14} style={{ color: 'var(--warning)' }} />
                <h3 className="font-semibold text-sm">Spearman 相关系数</h3>
              </div>
              <HeatMap matrix={edaResult.spearman_corr} />
            </div>

            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 size={14} style={{ color: 'var(--primary)' }} />
                <h3 className="font-semibold text-sm">特征重要性 (Random Forest)</h3>
              </div>
              <ImportanceChart data={edaResult.feature_importance} />
            </div>

            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <Zap size={14} style={{ color: 'var(--success)' }} />
                <h3 className="font-semibold text-sm">互信息</h3>
              </div>
              <ImportanceChart data={edaResult.mutual_info} />
            </div>
          </div>

          <div className="flex justify-between pt-2">
            <button onClick={onPrev} className="btn-secondary">
              <ArrowLeft size={14} /> 上一步
            </button>
            <button onClick={onNext} className="btn-primary">
              下一步：训练模型 <ArrowRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
