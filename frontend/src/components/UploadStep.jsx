import { useState, useCallback } from 'react';
import { Upload, FileSpreadsheet, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadFile } from '../api';

export default function UploadStep({ data, setData, onNext, error, setError }) {
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setError(null);
    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
      setError('仅支持 .xlsx / .xls 格式的 Excel 文件');
      return;
    }
    setLoading(true);
    try {
      const res = await uploadFile(file);
      setData(res.data);
    } catch (e) {
      setError('上传失败：' + (e.response?.data?.detail || e.message));
    }
    setLoading(false);
  }, [setData, setError]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="space-y-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold" style={{ color: 'var(--text)' }}>上传数据集</h2>
        <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
          上传 Excel 工作簿（.xlsx），选择工作表后开始分析
        </p>
      </div>

      {error && (
        <div className="error-banner flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById('file-input').click()}
        className="card cursor-pointer transition-colors"
        style={{
          padding: '48px 24px',
          textAlign: 'center',
          borderColor: dragOver ? 'var(--primary)' : undefined,
          background: dragOver ? 'rgba(37,99,235,0.02)' : undefined,
        }}
      >
        <input
          id="file-input"
          type="file"
          accept=".xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {loading ? (
          <div className="flex flex-col items-center gap-3">
            <div
              className="w-10 h-10 border-3 border-t-transparent rounded-full animate-spin"
              style={{ borderColor: 'var(--primary)', borderTopColor: 'transparent', borderWidth: 3 }}
            />
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>正在上传...</p>
          </div>
        ) : data ? (
          <div className="flex flex-col items-center gap-2">
            <CheckCircle size={40} style={{ color: 'var(--success)' }} />
            <p className="font-semibold" style={{ color: 'var(--success)' }}>
              {data.filename}
            </p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {data.rows} 行 x {data.columns} 列
            </p>
            <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
              点击或拖拽更换文件
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload size={36} style={{ color: 'var(--text-muted)' }} />
            <p className="font-medium">拖拽 Excel 文件到此处</p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>或点击选择文件</p>
          </div>
        )}
      </div>

      {data && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileSpreadsheet size={16} style={{ color: 'var(--primary)' }} />
            <h3 className="font-semibold text-sm">数据预览</h3>
            <span className="badge badge-primary">{data.rows} 行 x {data.columns} 列</span>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  {data.column_names.map((col) => (
                    <th key={col}>
                      {col}
                      <span className="block text-[10px] font-normal" style={{ color: 'var(--text-muted)' }}>
                        {data.column_types[col]}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.preview.slice(0, 10).map((row, i) => (
                  <tr key={i}>
                    {data.column_names.map((col) => (
                      <td key={col}>{row[col] ?? ''}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex justify-end">
            <button onClick={onNext} className="btn-primary">
              下一步：选择特征
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
