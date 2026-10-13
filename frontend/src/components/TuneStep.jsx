import { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { autoTune, getLLMStatus } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function TuneStep({ data, selection, trainResult, onPrev, error, setError }) {
  const [algorithm, setAlgorithm] = useState(trainResult?.algorithm || 'XGBoost');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [llmInfo, setLlmInfo] = useState(null);

  useEffect(() => {
    getLLMStatus().then((res) => setLlmInfo(res.data)).catch(() => {});
  }, []);

  const handleTune = async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const res = await autoTune({
        filename: data.filename,
        features: selection.features,
        target: selection.target,
        task_type: selection.taskType,
        algorithm,
      });
      setResult(res.data);
    } catch (e) {
      setError('调参失败：' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  };

  const chartData = result?.iterations?.map((it) => ({
    round: `第 ${it.round} 轮`,
    score: it.score,
  })) || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">LLM 智能调参</h2>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          大语言模型分析实验结果，提出新参数组合，后端真实训练验证
        </p>
      </div>

      {error && (
        <div className="error-banner flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Config */}
        <div className="card p-4 space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={14} style={{ color: 'var(--primary)' }} />
            <h3 className="font-semibold text-sm">调参配置</h3>
          </div>

          <div>
            <label className="text-xs mb-1 block" style={{ color: 'var(--text-secondary)' }}>算法</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="input-field text-sm"
            >
              {['Ridge', 'RandomForest', 'XGBoost', 'LightGBM', 'SVR', 'KNN', 'GBDT'].map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>

          {/* LLM status */}
          <div className="p-3 rounded-lg text-xs" style={{ background: 'var(--surface-alt)' }}>
            <div className="font-semibold mb-1" style={{ color: 'var(--text-secondary)' }}>LLM 连接状态</div>
            {llmInfo ? (
              <div className="space-y-1" style={{ color: 'var(--text-muted)' }}>
                <div>Provider: {llmInfo.provider || '未配置'}</div>
                <div>Model: {llmInfo.model || '-'}</div>
                <div>
                  状态: {llmInfo.configured ? (
                    <span style={{ color: 'var(--success)' }}>已配置</span>
                  ) : (
                    <span style={{ color: 'var(--warning)' }}>未配置 Key</span>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>加载中...</div>
            )}
          </div>

          <button
            onClick={handleTune}
            disabled={loading}
            className="btn-primary w-full justify-center"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'white', borderTopColor: 'transparent' }} />
            ) : (
              <>
                <Sparkles size={14} /> 开始调参
              </>
            )}
          </button>

          <div className="p-3 rounded-lg text-xs space-y-1.5" style={{ background: 'var(--surface-alt)' }}>
            <div className="font-semibold" style={{ color: 'var(--text-secondary)' }}>工作原理</div>
            <div style={{ color: 'var(--text-muted)' }}>
              1. LLM 分析已有实验结果和数据特征
            </div>
            <div style={{ color: 'var(--text-muted)' }}>
              2. 提出新参数组合及理由
            </div>
            <div style={{ color: 'var(--text-muted)' }}>
              3. 后端校验参数并真实训练
            </div>
            <div style={{ color: 'var(--text-muted)' }}>
              4. 将实测结果反馈给 LLM 继续优化
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2">
          {!result ? (
            <div className="card empty-state py-12">
              <Sparkles size={36} className="mx-auto mb-2" />
              <p className="text-sm">配置并启动调参</p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                LLM 将迭代优化超参数
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Score header */}
              <div className="card p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {result.reached_threshold ? (
                    <CheckCircle size={24} style={{ color: 'var(--success)' }} />
                  ) : (
                    <XCircle size={24} style={{ color: 'var(--warning)' }} />
                  )}
                  <div>
                    <div className="text-xl font-bold" style={{ color: 'var(--text)' }}>
                      {result.best_score?.toFixed(4)}
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {result.reached_threshold ? '已达到目标阈值' : '未达目标，已返回最佳结果'}
                    </div>
                  </div>
                </div>
                <span className="badge badge-primary">{result.iterations?.length || 0} 轮</span>
              </div>

              {/* Chart */}
              {chartData.length > 1 && (
                <div className="card p-4">
                  <h4 className="text-sm font-semibold mb-2">调参分数变化</h4>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E8EDF3" />
                      <XAxis dataKey="round" tick={{ fontSize: 11, fill: '#566477' }} />
                      <YAxis tick={{ fontSize: 11, fill: '#566477' }} />
                      <Tooltip
                        contentStyle={{ background: '#fff', border: '1px solid #DDE3EB', borderRadius: 6, fontSize: 12 }}
                      />
                      <Line type="monotone" dataKey="score" stroke="#2563EB" strokeWidth={2} dot={{ fill: '#2563EB', r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Iterations */}
              <div className="card p-4">
                <h4 className="text-sm font-semibold mb-2">迭代详情</h4>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {result.iterations?.map((it) => (
                    <div key={it.round} className="flex items-start gap-3 p-2.5 rounded-lg" style={{ background: 'var(--surface-alt)' }}>
                      <span className="text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0" style={{ background: 'rgba(37,99,235,0.1)', color: 'var(--primary)' }}>
                        {it.round}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                          分数: {it.score?.toFixed(4)}
                        </div>
                        {it.rationale && (
                          <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                            {it.rationale}
                          </div>
                        )}
                        <pre className="text-[11px] mt-1 overflow-x-auto" style={{ color: 'var(--text-muted)' }}>
                          {JSON.stringify(it.params)}
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Best params */}
              <div className="card p-4">
                <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--success)' }}>最佳参数</h4>
                <pre className="text-xs overflow-x-auto p-3 rounded-lg" style={{ background: 'var(--surface-alt)', color: 'var(--primary)' }}>
                  {JSON.stringify(result.best_params, null, 2)}
                </pre>
              </div>

              {/* Report */}
              {result.final_report && (
                <div className="card p-4">
                  <h4 className="text-sm font-semibold mb-2">调参报告</h4>
                  <pre className="text-xs whitespace-pre-wrap overflow-x-auto p-3 rounded-lg" style={{ background: 'var(--surface-alt)', color: 'var(--text-secondary)' }}>
                    {result.final_report}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-start pt-2">
        <button onClick={onPrev} className="btn-secondary">
          <ArrowLeft size={14} /> 上一步
        </button>
      </div>
    </div>
  );
}
