# mRModN 网页前端 / mRModN Web Frontend

## 项目背景 / Project Background

本项目是 mRModN RNA 修饰分类系统的**网页可视化前端**,提供多视图对比(定位对比、分类对比、柱状图对比等)与多模型注意力可视化能力,服务于研究人员的分析需求。

This project is the **web visualization frontend** of the mRModN RNA modification classification system. It provides multi-view comparisons (localization comparison, classification comparison, bar-chart comparison) and attention visualization across multiple models, serving researchers' analysis needs.

## 项目作用 / Purpose

**核心能力** / **Core capabilities**:
- 提交 RNA 序列触发后端推理
- Submit RNA sequences to trigger backend inference
- 展示 12 类 RNA 修饰位点的**定位可视化**(概率曲线 / 折线 / 柱状图)
- **Localization visualization** for 12 RNA modification types (probability curves / lines / bars)
- 多模型**对比视图**(mRModN vs EvoRMD 等)
- **Multi-model comparison view** (mRModN vs EvoRMD, etc.)
- 注意力权重热力图与 UMAP 嵌入
- Attention weight heatmaps and UMAP embeddings
- 完整的**中英双语**界面
- Full **Chinese + English bilingual** UI

## 技术栈 / Tech Stack

- React 19 + TypeScript
- Vite(构建工具,极快的冷启动)/ Vite (build tool with fast cold start)
- Ant Design 5(组件库)/ Ant Design 5 (UI library)
- TanStack Query(数据请求与缓存)/ TanStack Query (data fetching & cache)
- ECharts(图表:折线/柱状/饼图)/ ECharts (charts: line/bar/pie)
- @antv/g6(图可视化)/ @antv/g6 (graph viz)
- react-flow(流程图与节点编辑)/ react-flow (flow & node editor)
- ESLint 9(flat config)/ ESLint 9 (flat config)

## 目录结构 / Directory Layout

```
mRModN_WebAndWx_WebFrontend/
├── index.html              # HTML 入口
├── package.json            # 依赖与脚本
├── vite.config.ts          # Vite 构建配置(别名、插件、代理)
├── eslint.config.js        # ESLint flat config
├── tsconfig.json           # TS 根配置
├── tsconfig.app.json       # TS 应用配置
├── tsconfig.node.json      # TS Node 配置
├── .env.example            # 环境变量示例
├── config/
│   └── api.config.ts       # 后端 API 端点配置
├── public/                 # 静态资源(直接拷贝)
├── doc/                    # 项目文档
├── dist/                   # 构建产物(忽略)
├── src/
│   ├── pages/              # 主页面
│   │   ├── LocalizationViz.tsx        # 定位可视化主页
│   │   ├── LocComparisonViz.tsx       # 多模型定位对比
│   │   └── compare/
│   │       └── CompareBarChart.tsx    # 柱状图对比
│   ├── components/
│   │   └── workspace/      # 工作区组件
│   │       ├── types.ts   # 数据类型
│   │       └── mockData.ts # 模拟数据
│   └── lib/                # 工具库
│       ├── api.ts         # API 客户端
│       ├── uuidv7.ts      # UUID 生成
│       └── i18n/
│           ├── en.ts      # 英文文案
│           └── zh.ts      # 中文文案
└── .claude/                # Claude 配置(本地)
```

## 启动方式 / Getting Started

### 环境要求 / Requirements

- **Node.js ≥ 18**(推荐 20 LTS)/ Node.js ≥ 18 (recommend 20 LTS)
- npm 或 pnpm 或 yarn

### 安装 / Install

```bash
cd mRModN_WebAndWx_WebFrontend
npm install   # 或 pnpm install / yarn
```

### 配置 / Configure

```bash
cp .env.example .env
# 编辑 .env,设置后端地址 / edit .env to set backend URL
```

`vite.config.ts` 默认会代理 `/api` 到后端,无需额外配置。/ `vite.config.ts` proxies `/api` to backend by default.

### 启动开发服务器 / Start Dev Server

```bash
npm run dev
# 访问 http://localhost:5173 (或 Vite 输出端口)
# Visit http://localhost:5173 (or Vite's printed port)
```

### 构建生产版本 / Build for Production

```bash
npm run build        # 输出到 dist/
npm run preview      # 本地预览构建产物
```

### 代码检查 / Lint

```bash
npm run lint
```

## 关键文件说明 / Key Files

| 文件 / File | 作用 / Purpose |
|---|---|
| `config/api.config.ts` | 后端 API 端点配置(baseUrl、路径前缀)/ Backend API endpoint config (baseUrl, path prefixes) |
| `vite.config.ts` | Vite 构建配置:React 插件、路径别名(@/ → src/)、dev proxy、build 选项 / Vite config: React plugin, path aliases, dev proxy, build options |
| `eslint.config.js` | ESLint 9 flat config:TypeScript + React Hooks 规则 / ESLint 9 flat config with TS + React Hooks rules |
| `src/lib/api.ts` | API 客户端:基于 fetch(或 axios)的请求封装,统一错误处理与类型 / API client: fetch/axios wrapper, unified error handling & types |
| `src/lib/uuidv7.ts` | UUID v7(时间排序)生成器,用于任务 ID / UUID v7 (time-ordered) generator for task IDs |
| `src/lib/i18n/zh.ts` | 中文文案(模块化 key → 文案)/ Chinese i18n strings (modular key → text) |
| `src/lib/i18n/en.ts` | 英文文案 / English i18n strings |
| `src/pages/LocalizationViz.tsx` | 定位可视化主页:序列输入、12 类修饰概率曲线、注意力热图 / Localization viz main page: sequence input, 12-class probability curves, attention heatmap |
| `src/pages/LocComparisonViz.tsx` | 多模型定位对比:并排展示 mRModN / EvoRMD 等模型结果 / Multi-model localization comparison: side-by-side mRModN / EvoRMD results |
| `src/pages/compare/CompareBarChart.tsx` | 柱状图对比:不同模型 / 不同样本的指标对比 / Bar-chart comparison: metrics across models / samples |
| `src/components/workspace/types.ts` | 工作区数据模型:序列、预测、注意力、对比配置等 TS 类型 / Workspace data model: TS types for sequence, prediction, attention, comparison config |
| `src/components/workspace/mockData.ts` | 模拟数据:开发期前端不依赖后端时使用 / Mock data: for frontend dev without backend dependency |

## 国际化 / Internationalization

由 `src/lib/i18n/{zh,en}.ts` 提供模块化 key → 文案映射,运行时根据用户偏好切换。/ Provided by `src/lib/i18n/{zh,en}.ts` as a modular key → string map, switched at runtime by user preference.

## 与后端的对接 / Backend Integration

```
src/lib/api.ts → fetch('/api/...')
       ↓
vite.config.ts dev proxy → Cluster_WebAndWx_backend/server.py
       ↓
后端 → 异步任务 → 返回 batchJobId
       ↓
前端轮询 → 渲染到 LocalizationViz / LocComparisonViz
```

主要端点(详见 `config/api.config.ts`):/ Main endpoints (see `config/api.config.ts`):
- `POST /api/v1/submit-task` - 提交推理
- `GET  /api/v1/get-result?jobId=xxx` - 轮询结果
- `POST /api/v1/wx-login` - 登录(小程序/网页共享)

## 与其他项目的关系 / Relation to Other Projects

- **Cluster_WebAndWx_backend** - 后端 API,本项目通过 `src/lib/api.ts` 调用 / Backend API called via `src/lib/api.ts`
- **Cluster_WebAndWx_WxFrontend** - 微信小程序端,本项目是其网页版 / Web counterpart of the WeChat mini-program
- **mrmodn_sum** - 模型主项目,导出 ONNX 供后端加载 / Model project exporting ONNX for backend

## 注意事项 / Notes

- 开发期访问 `http://localhost:5173`,生产期通过 Nginx 等反向代理到后端 / In dev: localhost:5173; in prod: reverse-proxy to backend
- 大量图表渲染可能占用内存,生产构建前应开启代码分割(`vite.config.ts` → `build.rollupOptions.output.manualChunks`)/ Heavy charts may consume memory; enable code-splitting in prod build
- ESLint 9 flat config 与旧式 `.eslintrc` 不兼容 / ESLint 9 flat config is incompatible with legacy `.eslintrc`

## 相关论文 / Related Papers

参见 `/doc` 目录与项目主页。/ See `/doc` directory and project homepage.
