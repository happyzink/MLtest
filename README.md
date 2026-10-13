# MLtest · 科研机器学习工作台

基于 React 和 FastAPI 的中文机器学习实验项目，提供数据探索、模型训练、LLM 辅助调参和图表导出。项目持续开发中。

## 功能特性

- **中文浅色界面**：适配桌面和窄屏的响应式布局
- **模型训练**：提供 XGBoost、RandomForest、Ridge 等算法入口
- **散点图**：训练集/测试集真实值-预测值散点图，标注 R²/RMSE/MAE/n
- **图表导出**：Matplotlib 生成 PNG/SVG/PDF，支持中文，300 dpi
- **模型导出**：下载模型文件、参数和预测结果的 ZIP 包
- **LLM 调参**：通过后端 `.env` 配置 API Key，调用模型生成参数建议并执行训练

## 快速开始

### 1. 安装依赖

**后端：**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**前端：**
```bash
cd frontend
npm install
```

### 2. 配置 LLM（可选）

```bash
cd backend
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API 配置：
```dotenv
LLM_PROVIDER=openai_compatible
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

不配置 LLM 也可以使用普通训练功能。

### 3. 启动服务

**后端（终端 1）：**
```bash
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

**前端（终端 2）：**
```bash
cd frontend
npm run dev
```

访问 http://localhost:5173

## 使用流程

1. **上传数据**：上传 Excel 工作簿（.xlsx）
2. **选择特征**：选择目标变量和特征列，确定任务类型（回归/分类）
3. **探索分析**：查看相关性矩阵、特征重要性、互信息
4. **训练模型**：选择算法，查看训练集/测试集散点图和指标
5. **LLM 调参**：启动智能调参，LLM 分析结果并提出新参数

## 技术栈

- **前端**：React 19, Vite 8, Tailwind 4, Recharts
- **后端**：FastAPI, scikit-learn 1.5.2, XGBoost 2.1.1, LightGBM 4.5.0
- **可视化**：Matplotlib (论文图), Recharts (交互图)
- **存储**：joblib (模型), JSON (元数据), CSV (预测)

## 项目结构

```
ML-TEST/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # API 路由
│   │   ├── core/config.py   # 配置（加载 .env）
│   │   ├── services/
│   │   │   ├── ml_engine.py         # 训练引擎
│   │   │   ├── llm_client.py        # LLM 调用
│   │   │   ├── llm_tuner.py         # 调参循环
│   │   │   ├── artifact_service.py  # 模型保存
│   │   │   ├── plot_service.py      # 论文图表
│   │   │   └── report_service.py    # 比较报告
│   │   └── main.py
│   ├── .env.example
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/      # 页面组件
    │   ├── App.jsx
    │   └── index.css        # 浅色主题
    └── package.json
```

本地 `.env`、数据集、训练产物和开发交接材料不纳入版本控制。仓库仅提供不含密钥的 `.env.example` 配置模板。

## 开发命令

```bash
# 前端
npm run lint      # 代码检查
npm run build     # 构建
npm run dev       # 开发服务器

# 后端
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

## 许可证

MIT
