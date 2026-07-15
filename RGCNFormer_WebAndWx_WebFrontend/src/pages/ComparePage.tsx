/**
 * ComparePage.tsx - 多模型多指标对比页(综合面板)/ Multi-model comparison hub
 *
 * /compare 路由页面(独立 Layout,非 VizLayout)。综合展示多模型对比数据:
 *   - 柱状图(CompareBarChart):各模型 ACC/AUC/F1 等指标
 *   - mRModN 单模型热力图(MrmodnHeatmap)
 *   - 跨数据集热力图(DatasetComparisonHeatmap)
 *   - 单模型定位(LocalizationViz)
 *   - 跨模型定位对比(LocComparisonViz)
 *   - UMAP(UMapViz)
 * 数据由 useQuery 缓存;Layout 含"返回首页"按钮。
 * Page mounted at /compare (independent layout, not VizLayout). Aggregates multi-model
 * comparison data:
 *   - Bar chart (CompareBarChart): per-model ACC/AUC/F1
 *   - mRModN heatmap (MrmodnHeatmap)
 *   - Cross-dataset heatmap (DatasetComparisonHeatmap)
 *   - Single-model localization (LocalizationViz)
 *   - Cross-model localization (LocComparisonViz)
 *   - UMAP (UMapViz)
 * Data is useQuery-cached; the layout has a "Return to Home" button.
 *
 * 功能模块 / Modules:
 * - 6 种可视化子组件并列 / 6 sub-visualizations side by side
 * - useQuery 拉 6 类后端数据 / 6 useQuery hooks
 * - 独立 Layout(无 VizLayout Sider)/ Independent layout
 * - i18n 切换 / i18n switch
 *
 * 输入 / Inputs:
 * - URL: /compare(query 参数可指定对比模型集合,可选)
 * - 后端 /api/v1/model-comparison, dataset-comparison, mrmodn-heatmap,
 *   mrmodn-localization, mrmodn-loc-comparison, umap, cora-umap
 *
 * 输出 / Outputs:
 * - JSX.Element 综合对比页 / Comparison page JSX
 *
 * 数据流 / Data Flow:
 * 1. 页面挂载 → 6 个 useQuery 并发拉数据
 * 2. 各自把数据透传给子组件
 * 3. 子组件用 ECharts 渲染
 * 4. 用户滚动页面查看不同子视图
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(6 个 fetch 函数)、所有子 viz 组件
 * - 被调用 / Called by: App.tsx(<Route path="/compare">)
 * - 关联 / Related: VizDisplayPage.tsx(类似聚合页)
 *
 * 使用示例 / Usage Example:
 *   <Route path="/compare" element={<ComparePage />} />
 *   // 浏览器 /mrmodn/compare
 */
import React, { useState } from 'react';
import { Layout, Button } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { useQuery } from '@tanstack/react-query';
import { fetchModelComparison, fetchMrmodnHeatmap, fetchMrmodnLocalization, fetchMrmodnLocComparison, fetchUmapData, fetchCoraUmapData, fetchDatasetComparison } from '../lib/api';
import CompareBarChart from './compare/CompareBarChart';
import MrmodnHeatmap from './MrmodnHeatmap';
import DatasetComparisonHeatmap from './DatasetComparisonHeatmap';
import LocalizationViz from './LocalizationViz';
import LocComparisonViz from './LocComparisonViz';
import UMapViz from './UMapViz';
import AttentionComparisonViz from './AttentionComparisonViz';

const { Content, Sider } = Layout;

const DESKTOP_BREAKPOINT = 768;

type VizType = 'bar' | 'heatmap' | 'localization' | 'locComparison' | 'umap' | 'coraUmap' | 'datasetComparison' | 'attentionComparison';

const MORANDI = {
  sidebarBg: '#9E9288',
  sidebarLogoBg: '#8B7E74',
  sidebarText: '#F0EBE6',
  sidebarActive: '#B8A9C9',
  sidebarActiveText: '#FFFFFF',
  contentBg: '#F0EBE6',
  cardBg: '#FFFFFF',
  cardBorder: '#D4CEC6',
  titleColor: '#4A4A4A',
  titleUnderline: '#B8A9C9',
  btnBrown: '#C4A882',
  tableHeaderBg: '#9E9288',
  tableEvenRow: '#F5F0EB',
  tableHover: '#EBE5DE',
};

const ComparePage: React.FC = () => {
  const navigate = useNavigate();
  const { t, changeLanguage, language } = useTranslation();
  const [isDesktop, setIsDesktop] = useState(window.innerWidth > DESKTOP_BREAKPOINT);
  const [siderCollapsed, setSiderCollapsed] = useState(false);
  const [selectedVizType, setSelectedVizType] = useState<VizType>('bar');

  const { data: modelComparisonData, isLoading, isError } = useQuery({
    queryKey: ['modelComparison'],
    queryFn: fetchModelComparison,
  });

  const { data: heatmapData } = useQuery({
    queryKey: ['mrmodnHeatmap'],
    queryFn: fetchMrmodnHeatmap,
  });

  const { data: localizationData } = useQuery({
    queryKey: ['mrmodnLocalization'],
    queryFn: fetchMrmodnLocalization,
  });

  const { data: locComparisonData } = useQuery({
    queryKey: ['mrmodnLocComparison'],
    queryFn: fetchMrmodnLocComparison,
  });

  const { data: umapData } = useQuery({
    queryKey: ['umapData'],
    queryFn: fetchUmapData,
  });

  const { data: coraUmapData } = useQuery({
    queryKey: ['coraUmapData'],
    queryFn: fetchCoraUmapData,
  });

  const { data: datasetComparisonData } = useQuery({
    queryKey: ['datasetComparison'],
    queryFn: fetchDatasetComparison,
  });

  React.useEffect(() => {
    const handleResize = () => {
      const desktop = window.innerWidth > DESKTOP_BREAKPOINT;
      setIsDesktop(desktop);
      if (!desktop) {
        setSiderCollapsed(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          fontSize: 16,
          color: '#666'
        }}>
          {t('Loading comparison data...')}
        </div>
      );
    }

    if (isError) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          fontSize: 16,
          color: '#d32f2f'
        }}>
          {t('Failed to load comparison data')}
        </div>
      );
    }

    if (!modelComparisonData || modelComparisonData.models.length === 0) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          fontSize: 16,
          color: '#666'
        }}>
          No data available
        </div>
      );
    }

    switch (selectedVizType) {
      case 'bar':
        return <CompareBarChart data={modelComparisonData} />;
      case 'heatmap':
        return <MrmodnHeatmap data={heatmapData} />;
      case 'localization':
        return <LocalizationViz data={localizationData} />;
      case 'locComparison':
        return <LocComparisonViz data={locComparisonData} />;
      case 'umap':
        return <UMapViz data={umapData} />;
      case 'coraUmap':
        return <UMapViz data={coraUmapData} />;
      case 'datasetComparison':
        return <DatasetComparisonHeatmap data={datasetComparisonData} />;
      case 'attentionComparison':
        return <AttentionComparisonViz />;
      default:
        return null;
    }
  };

  const renderSummaryTable = () => {
    if (!modelComparisonData || modelComparisonData.models.length === 0 || isLoading || isError) return null;

    return (
      <table style={{
        width: '100%',
        borderCollapse: 'collapse',
        marginTop: 24,
      }}>
        <thead>
          <tr>
            {['Model Name', ...modelComparisonData.metric_names].map((col) => (
              <th
                key={col}
                style={{
                  backgroundColor: MORANDI.tableHeaderBg,
                  color: '#FFFFFF',
                  padding: '8px 12px',
                  textAlign: 'center',
                  fontWeight: 600,
                  border: `1px solid ${MORANDI.cardBorder}`,
                }}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {modelComparisonData.models.map((model, rowIdx) => (
            <tr
              key={model.name}
              style={{
                backgroundColor: rowIdx % 2 === 0 ? MORANDI.tableEvenRow : MORANDI.cardBg,
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = MORANDI.tableHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = rowIdx % 2 === 0 ? MORANDI.tableEvenRow : MORANDI.cardBg)}
            >
              <td style={{
                padding: '6px 12px',
                textAlign: 'left',
                fontWeight: 600,
                color: MORANDI.titleColor,
                border: `1px solid ${MORANDI.cardBorder}`,
              }}>
                {model.display_name}
              </td>
              {modelComparisonData.metric_names.map((metricName) => (
                <td
                  key={metricName}
                  style={{
                    padding: '6px 12px',
                    textAlign: 'center',
                    color: MORANDI.titleColor,
                    border: `1px solid ${MORANDI.cardBorder}`,
                  }}
                >
                  {(model.metrics[metricName] * 100).toFixed(1)}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const cardBase = (isSelected: boolean) => ({
    borderRadius: 16,
    padding: '12px 16px',
    backgroundColor: isSelected ? MORANDI.sidebarActive : 'transparent',
    cursor: 'pointer' as const,
    display: 'flex' as const,
    alignItems: 'center' as const,
    gap: 8,
    margin: '0 16px',
    color: isSelected ? MORANDI.sidebarActiveText : MORANDI.sidebarText,
    fontWeight: 500 as const,
  });

  const desktopSidebar = (
    <div style={{ flexGrow: 1 }}>
      <div style={{
        height: 'auto',
        minHeight: 32,
        margin: 16,
        backgroundColor: MORANDI.sidebarLogoBg,
        borderRadius: 8,
        padding: '12px 8px',
        color: '#fff',
        fontWeight: 'bold',
        textAlign: 'center',
        wordBreak: 'break-word',
        overflow: 'hidden',
      }}>
        {siderCollapsed ? 'RNA' : t('Model Performance Comparison')}
      </div>

      <div
        onClick={() => setSelectedVizType('heatmap')}
        style={{
          ...cardBase(selectedVizType === 'heatmap'),
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🔬</span>
        {!siderCollapsed && <span>{t('DCPRES Classification')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('datasetComparison')}
        style={{
          ...cardBase(selectedVizType === 'datasetComparison'),
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>📋</span>
        {!siderCollapsed && <span>{t('Dataset Comparison')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('localization')}
        style={{
          ...cardBase(selectedVizType === 'localization'),
          marginTop: 4,
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🎯</span>
        {!siderCollapsed && <span>{t('Localization')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('locComparison')}
        style={{
          ...cardBase(selectedVizType === 'locComparison'),
          marginTop: 4,
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🏆</span>
        {!siderCollapsed && <span>{t('Loc Comparison')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('umap')}
        style={{
          ...cardBase(selectedVizType === 'umap'),
          marginTop: 4,
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🗺️</span>
        {!siderCollapsed && <span>{t('UMAP Visualization')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('coraUmap')}
        style={{
          ...cardBase(selectedVizType === 'coraUmap'),
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🌐</span>
        {!siderCollapsed && <span>{t('CORA UMAP Visualization')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('bar')}
        style={{
          ...cardBase(selectedVizType === 'bar'),
          marginTop: 4,
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>📊</span>
        {!siderCollapsed && <span>{t('Bar Chart')}</span>}
      </div>

      <div
        onClick={() => setSelectedVizType('attentionComparison')}
        style={{
          ...cardBase(selectedVizType === 'attentionComparison'),
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 16 }}>🧠</span>
        {!siderCollapsed && <span>{t('Attention Comparison')}</span>}
      </div>
    </div>
  );

  const mobileCards = (
    <>
      <div
        onClick={() => setSelectedVizType('heatmap')}
        style={{
          ...cardBase(selectedVizType === 'heatmap'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>🔬</span>
        <span>{t('DCPRES Classification')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('datasetComparison')}
        style={{
          ...cardBase(selectedVizType === 'datasetComparison'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>📋</span>
        <span>{t('Dataset Comparison')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('localization')}
        style={{
          ...cardBase(selectedVizType === 'localization'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>🎯</span>
        <span>{t('Localization')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('locComparison')}
        style={{
          ...cardBase(selectedVizType === 'locComparison'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>🏆</span>
        <span>{t('Loc Comparison')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('umap')}
        style={{
          ...cardBase(selectedVizType === 'umap'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>🗺️</span>
        <span>{t('UMAP Visualization')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('coraUmap')}
        style={{
          ...cardBase(selectedVizType === 'coraUmap'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>🌐</span>
        <span>{t('CORA UMAP Visualization')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('bar')}
        style={{
          ...cardBase(selectedVizType === 'bar'),
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 16 }}>📊</span>
        <span>{t('Bar Chart')}</span>
      </div>
      <div
        onClick={() => setSelectedVizType('attentionComparison')}
        style={{
          ...cardBase(selectedVizType === 'attentionComparison'),
        }}
      >
        <span style={{ fontSize: 16 }}>🧠</span>
        <span>{t('Attention Comparison')}</span>
      </div>
    </>
  );

  if (isDesktop) {
    return (
      <Layout style={{ minHeight: '100vh', backgroundColor: MORANDI.contentBg }}>
        <Sider
          collapsible
          collapsed={siderCollapsed}
          onCollapse={setSiderCollapsed}
          width={200}
          style={{
            overflow: 'auto',
            height: '100vh',
            position: 'fixed',
            left: 0,
            top: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: MORANDI.sidebarBg,
          }}
        >
          {desktopSidebar}
          <div style={{ padding: '16px', textAlign: 'center' }}>
            <Button
              type="primary"
              onClick={() => navigate('/nextgen')}
              style={{ marginBottom: '16px', width: '100%', backgroundColor: MORANDI.btnBrown, borderColor: MORANDI.btnBrown }}
            >
              {t('Return to Home')}
            </Button>
            <div>
              <Button
                onClick={() => changeLanguage('zh')}
                disabled={language === 'zh'}
                size="small"
                style={{ marginRight: 8 }}
              >
                中文
              </Button>
              <Button
                onClick={() => changeLanguage('en')}
                disabled={language === 'en'}
                size="small"
              >
                EN
              </Button>
            </div>
          </div>
        </Sider>
        <Layout style={{ marginLeft: siderCollapsed ? 80 : 200, transition: 'all 0.2s', height: '100vh', overflow: 'hidden' }}>
          <Content style={{
            padding: 20,
            overflow: 'auto',
            height: '100%',
            backgroundColor: MORANDI.contentBg,
          }}>
            <h1 style={{
              fontSize: 24,
              fontWeight: 'bold',
              marginBottom: 16,
              color: MORANDI.titleColor,
              borderBottom: `2px solid ${MORANDI.titleUnderline}`,
              paddingBottom: 8,
            }}>
              {t('Model Performance Comparison')}
            </h1>
            <div style={{
              backgroundColor: MORANDI.cardBg,
              borderRadius: 12,
              padding: 24,
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            }}>
              {renderContent()}
            </div>
            {selectedVizType === 'bar' && renderSummaryTable()}
          </Content>
        </Layout>
      </Layout>
    );
  }

  return (
    <Layout style={{ height: '100vh', backgroundColor: MORANDI.contentBg, overflow: 'hidden' }}>
      <div style={{
        backgroundColor: MORANDI.sidebarBg,
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div style={{
          color: '#fff',
          fontWeight: 'bold',
          fontSize: 16,
          wordBreak: 'break-word',
          maxWidth: '70%',
        }}>
          {t('Model Performance Comparison')}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button
            size="small"
            onClick={() => changeLanguage('zh')}
            disabled={language === 'zh'}
            style={{ color: language === 'zh' ? MORANDI.sidebarActive : '#fff', borderColor: 'transparent', background: 'transparent' }}
          >
            中文
          </Button>
          <Button
            size="small"
            onClick={() => changeLanguage('en')}
            disabled={language === 'en'}
            style={{ color: language === 'en' ? MORANDI.sidebarActive : '#fff', borderColor: 'transparent', background: 'transparent' }}
          >
            EN
          </Button>
          <Button size="small" type="primary" onClick={() => navigate('/nextgen')} style={{ backgroundColor: MORANDI.btnBrown, borderColor: MORANDI.btnBrown }}>
            {t('Return to Home')}
          </Button>
        </div>
      </div>
      <Content style={{
        padding: 16,
        overflow: 'auto',
        flex: 1,
      }}>
        {mobileCards}
        <div style={{
          backgroundColor: MORANDI.cardBg,
          borderRadius: 12,
          padding: 16,
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        }}>
          {renderContent()}
        </div>
        {selectedVizType === 'bar' && renderSummaryTable()}
      </Content>
    </Layout>
  );
};

export default ComparePage;