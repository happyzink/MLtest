import { useState } from 'react';
import { Menu, X, Database, Settings } from 'lucide-react';

const STEP_LABELS = [
  '上传数据',
  '特征与划分',
  '探索分析',
  '训练模型',
  'LLM 调参',
];

export default function AppShell({
  currentStep,
  onStepClick,
  completedSteps,
  llmStatus,
  datasetInfo,
  children,
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const canClick = (i) => i <= Math.max(...completedSteps, 0) + 1;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--surface-alt)' }}>
      {/* Top bar */}
      <header
        className="h-14 flex items-center justify-between px-4 lg:px-6 border-b shrink-0"
        style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-3">
          <button
            className="lg:hidden p-1.5 rounded-md"
            style={{ border: '1px solid var(--border)' }}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
          <h1 className="text-base font-bold" style={{ color: 'var(--primary)' }}>
            ML-TEST
          </h1>
          <span className="text-xs hidden sm:inline" style={{ color: 'var(--text-muted)' }}>
            科研机器学习工作台
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          {datasetInfo && (
            <span className="hidden sm:flex items-center gap-1">
              <Database size={12} />
              {datasetInfo}
            </span>
          )}
          <span className="flex items-center gap-1">
            <Settings size={12} />
            LLM: {llmStatus || '未配置'}
          </span>
        </div>
      </header>

      <div className="flex flex-1 min-w-0">
        {/* Sidebar - desktop */}
        <nav
          className="hidden lg:flex flex-col w-56 shrink-0 border-r p-3 gap-1 overflow-y-auto"
          style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          {STEP_LABELS.map((label, i) => {
            const active = i === currentStep;
            const completed = completedSteps.includes(i);
            const disabled = !canClick(i);
            return (
              <button
                key={i}
                className={`sidebar-item ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${disabled ? 'disabled' : ''}`}
                onClick={() => !disabled && onStepClick(i)}
                disabled={disabled}
              >
                <span className="step-number">{completed ? '✓' : i + 1}</span>
                {label}
              </button>
            );
          })}
        </nav>

        {/* Sidebar - mobile overlay */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 z-40"
            onClick={() => setSidebarOpen(false)}
          >
            <div className="absolute inset-0 bg-black/20" />
            <nav
              className="absolute left-0 top-14 bottom-0 w-56 p-3 gap-1 flex flex-col border-r overflow-y-auto"
              style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
              onClick={(e) => e.stopPropagation()}
            >
              {STEP_LABELS.map((label, i) => {
                const active = i === currentStep;
                const completed = completedSteps.includes(i);
                const disabled = !canClick(i);
                return (
                  <button
                    key={i}
                    className={`sidebar-item ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${disabled ? 'disabled' : ''}`}
                    onClick={() => {
                      if (!disabled) {
                        onStepClick(i);
                        setSidebarOpen(false);
                      }
                    }}
                    disabled={disabled}
                  >
                    <span className="step-number">{completed ? '✓' : i + 1}</span>
                    {label}
                  </button>
                );
              })}
            </nav>
          </div>
        )}

        {/* Mobile step tabs */}
        <div
          className="lg:hidden fixed top-14 left-0 right-0 z-30 flex overflow-x-auto border-b shrink-0"
          style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          {STEP_LABELS.map((label, i) => {
            const active = i === currentStep;
            return (
              <button
                key={i}
                className="flex-shrink-0 px-3 py-2 text-xs font-medium border-b-2 transition-colors"
                style={{
                  color: active ? 'var(--primary)' : 'var(--text-muted)',
                  borderColor: active ? 'var(--primary)' : 'transparent',
                }}
                onClick={() => canClick(i) && onStepClick(i)}
              >
                {i + 1}. {label}
              </button>
            );
          })}
        </div>

        {/* Main content */}
        <main className="flex-1 min-w-0 overflow-y-auto p-4 lg:p-6 mt-10 lg:mt-0">
          <div className="max-w-[1400px] mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

export { STEP_LABELS };
