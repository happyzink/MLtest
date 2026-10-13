import { useState } from 'react';
import { ArrowLeft, ArrowRight, Columns, Target, Cpu, Search, AlertCircle } from 'lucide-react';

export default function FeatureStep({ data, selection, setSelection, onNext, onPrev, error, setError }) {
  const [features, setFeatures] = useState(selection?.features || []);
  const [target, setTarget] = useState(selection?.target || '');
  const [taskType, setTaskType] = useState(selection?.taskType || 'regression');
  const [search, setSearch] = useState('');

  const cols = data?.column_names || [];
  const colTypes = data?.column_types || {};

  const toggleFeature = (col) => {
    setFeatures((prev) =>
      prev.includes(col) ? prev.filter((f) => f !== col) : [...prev, col]
    );
  };

  const selectAllNumeric = () => {
    const numericCols = cols.filter(
      (c) => colTypes[c] === 'numeric' && c !== target
    );
    setFeatures(numericCols);
  };

  const clearFeatures = () => setFeatures([]);

  const handleTargetChange = (col) => {
    setTarget(col);
    setFeatures((prev) => prev.filter((f) => f !== col));
  };

  const handleNext = () => {
    setError(null);
    if (features.length === 0) return setError('请至少选择一个特征');
    if (taskType !== 'unsupervised' && !target) return setError('请选择目标变量');
    if (taskType !== 'unsupervised' && features.includes(target)) {
      return setError('目标变量不能同时作为特征');
    }
    setSelection({ features, target, taskType });
    onNext();
  };

  const filteredCols = cols.filter((c) =>
    c.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">配置机器学习任务</h2>
          <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
            选择特征（X）、目标变量（Y）和任务类型
          </p>
        </div>
      </div>

      {error && (
        <div className="error-banner flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Task Type */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Cpu size={16} style={{ color: 'var(--primary)' }} />
            <h3 className="font-semibold text-sm">任务类型</h3>
          </div>
          {[
            { value: 'regression', label: '回归 (Regression)' },
            { value: 'classification', label: '分类 (Classification)' },
            { value: 'unsupervised', label: '无监督 (Unsupervised)' },
          ].map((t) => (
            <button
              key={t.value}
              onClick={() => setTaskType(t.value)}
              className="w-full text-left px-3 py-2.5 rounded-lg mb-1.5 text-sm font-medium transition-colors"
              style={{
                background: taskType === t.value ? 'rgba(37,99,235,0.08)' : 'transparent',
                color: taskType === t.value ? 'var(--primary)' : 'var(--text-secondary)',
                border: taskType === t.value ? '1px solid rgba(37,99,235,0.2)' : '1px solid transparent',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Features */}
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Columns size={16} style={{ color: 'var(--primary)' }} />
              <h3 className="font-semibold text-sm">特征 (X)</h3>
              <span className="badge badge-primary">{features.length} 个</span>
            </div>
            <div className="flex gap-1">
              <button onClick={selectAllNumeric} className="text-xs px-2 py-1 rounded" style={{ color: 'var(--primary)', background: 'rgba(37,99,235,0.06)' }}>
                全选数值
              </button>
              <button onClick={clearFeatures} className="text-xs px-2 py-1 rounded" style={{ color: 'var(--text-muted)', background: 'var(--surface-alt)' }}>
                清空
              </button>
            </div>
          </div>

          <div className="relative mb-2">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索列名..."
              className="input-field text-sm pl-8"
            />
          </div>

          <div className="max-h-64 overflow-y-auto space-y-0.5">
            {filteredCols.filter((c) => c !== target).map((col) => {
              const selected = features.includes(col);
              return (
                <button
                  key={col}
                  onClick={() => toggleFeature(col)}
                  className="w-full text-left px-2.5 py-1.5 rounded text-sm flex items-center justify-between transition-colors"
                  style={{
                    background: selected ? 'rgba(37,99,235,0.06)' : 'transparent',
                  }}
                >
                  <span style={{ color: selected ? 'var(--primary)' : 'var(--text)', fontSize: 13 }}>
                    {col}
                  </span>
                  <span className="text-[10px] px-1.5 rounded" style={{ background: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
                    {colTypes[col]}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Target */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Target size={16} style={{ color: 'var(--accent)' }} />
            <h3 className="font-semibold text-sm">目标变量 (Y)</h3>
          </div>
          {taskType === 'unsupervised' ? (
            <p className="text-sm p-3 rounded-lg" style={{ background: 'rgba(37,99,235,0.04)', color: 'var(--text-secondary)' }}>
              无监督学习不需要目标变量
            </p>
          ) : (
            <div className="max-h-72 overflow-y-auto space-y-0.5">
              {filteredCols.map((col) => (
                <button
                  key={col}
                  onClick={() => handleTargetChange(col)}
                  className="w-full text-left px-2.5 py-1.5 rounded text-sm transition-colors"
                  style={{
                    background: target === col ? 'rgba(37,99,235,0.08)' : 'transparent',
                    color: target === col ? 'var(--primary)' : 'var(--text)',
                    fontWeight: target === col ? 600 : 400,
                  }}
                >
                  {col}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-between pt-2">
        <button onClick={onPrev} className="btn-secondary">
          <ArrowLeft size={14} /> 上一步
        </button>
        <button onClick={handleNext} className="btn-primary">
          下一步：探索分析 <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
