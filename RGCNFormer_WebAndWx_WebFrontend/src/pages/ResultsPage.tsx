/**
 * ResultsPage.tsx - 单任务结果聚合页(由 :jobId 驱动)/ Per-job result aggregator
 *
 * /results/:jobId 路由页面。接收 URL 中的 jobId,useQuery 拉取该任务完整结果
 * (classification / attention / gcn / ig / umap),并以左右分栏 / Tab 形式聚合展示
 * 所有可视化子组件。配合后端的"提交任务 + 轮询"流程使用。
 * Page mounted at /results/:jobId. Reads :jobId from the URL, fetches the full
 * result (classification/attention/gcn/ig/umap) via useQuery, and renders every
 * visualization sub-component in a sidebar-tab layout. Used after backend submit+polling.
 *
 * 功能模块 / Modules:
 * - :jobId URL 参数解析 / :jobId URL param parsing
 * - useQuery 拉任务结果(fetchResult by jobId)
 * - Tab 菜单(分类/注意力/GCN/IG/UMAP/对比)
 *   / Tab menu (classification/attention/GCN/IG/UMAP/compare)
 * - 选中 Tab → 渲染对应子组件 / Selected tab → render child
 *
 * 输入 / Inputs:
 * - URL params: { jobId: string } / job id from URL
 * - 后端 /api/v1/get-result?jobId=xxx 返回 ResultData
 *
 * 输出 / Outputs:
 * - JSX.Element 结果聚合页 / Results page JSX
 *
 * 数据流 / Data Flow:
 * 1. URL 进入 /results/:jobId
 * 2. useParams 拿 jobId → useQuery(fetchResult, jobId)
 * 3. 拿到 ResultData → 渲染 Tab 菜单
 * 4. 用户切 Tab → 渲染对应子组件
 * 5. 任何子组件用 lib/api 进一步拉 viz 数据
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchResult by jobId)、多个 viz 子组件
 * - 被调用 / Called by: App.tsx(<Route path="/results/:jobId">)
 * - 关联 / Related: VizDisplayPage.tsx(类似聚合,但接收 state 而非 URL)
 *
 * 使用示例 / Usage Example:
 *   <Route path="/results/:jobId" element={<ResultsPage />} />
 *   // 浏览器 /mrmodn/results/<uuid>
 */
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { MenuOutlined, ClusterOutlined, EyeOutlined, PartitionOutlined, LineChartOutlined, AimOutlined, BarChartOutlined } from '@ant-design/icons';
import ClassificationViz from './ClassificationViz';
import AttentionViz from './AttentionViz';
import AttentionDistributionViz from './AttentionDistributionViz';
import GcnViz from './GcnViz';
import TargetGcnViz from './TargetGcnViz';
import IntegratedGradientsViz from './IntegratedGradientsViz';
import { fetchResult, isProcessing, isCompleted, isFailed, getErrorMessage } from '../lib/api';
import { useTranslation } from '../lib/i18n/LanguageContext';
import './ResultsPage.css';

type ViewType = 'classification' | 'attention' | 'attention-score' | 'gcn' | 'target-gcn' | 'integrated-gradients' | 'model-viz';

const POLL_INTERVAL = 3000; // 3 seconds
const TIMEOUT = 300000; // 5 minutes

const ResultsPage: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [activeView, setActiveView] = useState<ViewType>('classification');
  const [isMobile, setIsMobile] = useState(false);
  const [hasTimedOut, setHasTimedOut] = useState(false);
  const { t } = useTranslation();

  // Detect mobile/desktop
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // TanStack Query for polling results
  const {
    data: resultData,
    status,
    error,
    isError,
  } = useQuery({
    queryKey: ['result', jobId],
    queryFn: () => fetchResult(jobId!),
    enabled: !!jobId && !hasTimedOut,
    refetchInterval(data) {
      // Only poll if still processing and hasn't timed out
      if (hasTimedOut) return false;
      // Check if data indicates processing status
      if (!data) return POLL_INTERVAL;
      return data.state.data?.status === 'processing' ? POLL_INTERVAL : false;
    },
    refetchIntervalInBackground: true,
  });

  // Handle timeout
  useEffect(() => {
    if (!jobId) return;

    const timeoutTimer = setTimeout(() => {
      if (isProcessing(resultData)) {
        setHasTimedOut(true);
      }
    }, TIMEOUT);

    return () => clearTimeout(timeoutTimer);
  }, [jobId, resultData]);

  // Loading state (polling)
  if (status === 'pending' || (isProcessing(resultData) && !hasTimedOut)) {
    return (
      <div className="results-page loading">
        <div className="loading-spinner"></div>
        <p>{t('Results calculation in progress, please wait...')}</p>
        <p className="job-id" style={{ marginTop: '10px', fontSize: '14px', color: '#999' }}>
          {t('Task ID:')} {jobId}
        </p>
      </div>
    );
  }

  // Error state
  if (isError || isFailed(resultData)) {
    const errorMessage = getErrorMessage(resultData) || (error as Error)?.message || t('Unable to load data');
    return (
      <div className="results-page error">
        <div className="error-icon">⚠️</div>
        <h2>{t('Processing failed')}</h2>
        <p>{errorMessage}</p>
        <p className="job-id" style={{ marginTop: '10px', fontSize: '14px', color: '#999' }}>
          {t('Task ID:')} {jobId}
        </p>
      </div>
    );
  }

  // Timeout state
  if (hasTimedOut) {
    return (
      <div className="results-page error">
        <div className="error-icon">⏱️</div>
        <h2>{t('Processing timeout')}</h2>
        <p>{t('Task processing timeout, please try again later')}</p>
        <p className="job-id" style={{ marginTop: '10px', fontSize: '14px', color: '#999' }}>
          {t('Task ID:')} {jobId}
        </p>
      </div>
    );
  }

  // No data state
  if (!resultData || !isCompleted(resultData)) {
    return (
      <div className="results-page error">
        <div className="error-icon">⚠️</div>
        <h2>{t('No data')}</h2>
        <p>{t('Unable to load visualization data')}</p>
      </div>
    );
  }

  const menuItems = [
    {
      key: 'classification',
      label: t('Classification Tree'),
      icon: <ClusterOutlined />,
      component: <ClassificationViz data={resultData.classification} />
    },
    {
      key: 'attention',
      label: t('Attention'),
      icon: <EyeOutlined />,
      component: <AttentionViz data={resultData.attention} />
    },
    {
      key: 'attention-score',
      label: t('Attention Score'),
      icon: <BarChartOutlined />,
      component: <AttentionDistributionViz />
    },
    {
      key: 'gcn',
      label: t('GCN Graph'),
      icon: <PartitionOutlined />,
      component: <GcnViz data={resultData.gcn} />
    },
    {
      key: 'target-gcn',
      label: t('GCN Message Passing'),
      icon: <AimOutlined />,
      component: <TargetGcnViz targetNodeIdx={0} />
    },
    {
      key: 'integrated-gradients',
      label: t('Integrated Gradients'),
      icon: <LineChartOutlined />,
      component: <IntegratedGradientsViz />
    },
    // {
    //   key: 'model-viz',
    //   label: t('Model Architecture'),
    //   icon: <ApartmentOutlined />,
    //   component: <ModelViz />
    // }
  ];

  return (
    <div className="results-page">
      <div className="results-header">
        <h1>{t('RNA Sequence Analysis Results')}</h1>
        <p className="job-id">{t('Task ID:')} {jobId}</p>
      </div>

      <div className="results-layout">
        {!isMobile && (
          <div className="sidebar">
            <div className="sidebar-header">
              <MenuOutlined />
              <span>{t('Visualization Options')}</span>
            </div>
            <div className="sidebar-menu">
              {menuItems.map(item => (
                <button
                  key={item.key}
                  className={`sidebar-item ${activeView === item.key ? 'active' : ''}`}
                  onClick={() => setActiveView(item.key as ViewType)}
                >
                  <span className="sidebar-item-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="results-content">
          <div className="content-wrapper">
            {menuItems.find(item => item.key === activeView)?.component}
          </div>
        </div>

        {isMobile && (
          <div className="mobile-tabs">
            {menuItems.map(item => (
              <button
                key={item.key}
                className={`mobile-tab ${activeView === item.key ? 'active' : ''}`}
                onClick={() => setActiveView(item.key as ViewType)}
              >
                <span className="mobile-tab-icon">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
