import { useState, useEffect } from 'react';
import AppShell from './components/layout/AppShell';
import UploadStep from './components/UploadStep';
import FeatureStep from './components/FeatureStep';
import EDAStep from './components/EDAStep';
import TrainStep from './components/TrainStep';
import TuneStep from './components/TuneStep';
import { getLLMStatus } from './api';

export default function App() {
  const [step, setStep] = useState(0);
  const [data, setData] = useState(null);
  const [selection, setSelection] = useState(null);
  const [edaResult, setEdaResult] = useState(null);
  const [trainResult, setTrainResult] = useState(null);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [error, setError] = useState(null);
  const [llmStatus, setLlmStatus] = useState('加载中...');

  useEffect(() => {
    getLLMStatus()
      .then((res) => {
        const info = res.data;
        if (info.configured) {
          setLlmStatus(`${info.provider} (${info.model})`);
        } else {
          setLlmStatus('未配置');
        }
      })
      .catch(() => setLlmStatus('连接失败'));
  }, []);

  const markComplete = (s) => {
    setCompletedSteps((prev) => (prev.includes(s) ? prev : [...prev, s]));
  };

  const next = () => {
    markComplete(step);
    setStep((s) => Math.min(s + 1, 4));
    setError(null);
  };
  const prev = () => {
    setStep((s) => Math.max(s - 1, 0));
    setError(null);
  };

  const datasetInfo = data ? `${data.filename} (${data.rows}x${data.columns})` : null;

  return (
    <AppShell
      currentStep={step}
      onStepClick={setStep}
      completedSteps={completedSteps}
      llmStatus={llmStatus}
      datasetInfo={datasetInfo}
    >
      {step === 0 && (
        <UploadStep
          data={data}
          setData={(d) => { setData(d); setEdaResult(null); setTrainResult(null); }}
          onNext={next}
          error={error}
          setError={setError}
        />
      )}
      {step === 1 && (
        <FeatureStep
          data={data}
          selection={selection}
          setSelection={(s) => { setSelection(s); setEdaResult(null); setTrainResult(null); }}
          onNext={next}
          onPrev={prev}
          error={error}
          setError={setError}
        />
      )}
      {step === 2 && (
        <EDAStep
          data={data}
          selection={selection}
          edaResult={edaResult}
          setEdaResult={setEdaResult}
          onNext={next}
          onPrev={prev}
          error={error}
          setError={setError}
        />
      )}
      {step === 3 && (
        <TrainStep
          data={data}
          selection={selection}
          trainResult={trainResult}
          setTrainResult={setTrainResult}
          onNext={next}
          onPrev={prev}
          error={error}
          setError={setError}
        />
      )}
      {step === 4 && (
        <TuneStep
          data={data}
          selection={selection}
          trainResult={trainResult}
          onPrev={prev}
          error={error}
          setError={setError}
        />
      )}
    </AppShell>
  );
}
