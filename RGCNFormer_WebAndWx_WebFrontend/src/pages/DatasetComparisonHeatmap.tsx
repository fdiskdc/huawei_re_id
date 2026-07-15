/**
 * DatasetComparisonHeatmap.tsx - 跨数据集分类结果热力图 / Cross-dataset classification heatmap
 *
 * /viz-display 之一(也可独立访问)。使用 ECharts 渲染一个二维热力图,行 = 12 类
 * RNA 修饰,列 = 多个数据集(Human / Plant / 3Gen / ac4C 等),颜色深浅表示
 * 该模型在该数据集该修饰上的分类性能(如 ACC / AUC)。
 * 数据由 fetchDatasetComparison(GET)获取,经 useQuery 缓存。
 * One of the /viz-display pages. Renders an ECharts heatmap: rows = 12 RNA modification
 * types, columns = multiple datasets (Human/Plant/3Gen/ac4C/etc.), color intensity =
 * per-modification classification metric (ACC/AUC). Data via fetchDatasetComparison +
 * useQuery caching.
 *
 * 功能模块 / Modules:
 * - ECharts 热力图(visualMap 渐变)/ ECharts heatmap with visualMap gradient
 * - 行/列 tooltip(显示具体指标值)/ Row/column tooltip
 * - 莫兰迪色卡(MORANDI 常量)/ Morandi palette
 * - 加载/错误态 / Loading/error states
 *
 * 输入 / Inputs:
 * - 后端 /api/v1/dataset-comparison 返回 DatasetComparisonData
 *   / Backend /api/v1/dataset-comparison returns DatasetComparisonData
 *
 * 输出 / Outputs:
 * - JSX.Element 热力图 / Heatmap JSX
 *
 * 数据流 / Data Flow:
 * 1. useQuery → fetchDatasetComparison()
 * 2. 把 data 转换为 [x, y, value] 三元组数组
 * 3. 配置 ECharts option(series.type='heatmap', visualMap)
 * 4. echart.setOption(option) → 渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchDatasetComparison, DatasetComparisonData 类型)
 * - 被调用 / Called by: App.tsx(<Route path="/viz-display"> 内部使用)
 * - 关联 / Related: MrmodnHeatmap.tsx(单数据集热力图)
 *
 * 使用示例 / Usage Example:
 *   const { data, isLoading, error } = useQuery({
 *     queryKey: ['dataset-comparison'],
 *     queryFn: fetchDatasetComparison,
 *   });
 */
import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import type { ECharts } from 'echarts';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { useQuery } from '@tanstack/react-query';
import { fetchDatasetComparison, type DatasetComparisonData } from '../lib/api';

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
};

const HEATMAP_COLORS = ['#EDE3DA', '#B8A9C9', '#8DA399'];
const NULL_COLOR = '#D4CEC6';

interface DatasetComparisonHeatmapProps {
  data?: DatasetComparisonData;
}

const DatasetComparisonHeatmap: React.FC<DatasetComparisonHeatmapProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<ECharts | null>(null);

  const { data: queryData, isLoading, isError } = useQuery({
    queryKey: ['datasetComparison'],
    queryFn: fetchDatasetComparison,
    enabled: !propData,
  });

  const data = propData || queryData;

  useEffect(() => {
    if (!chartRef.current || !data) return;

    const container = chartRef.current;
    const isMobile = window.innerWidth < 768;

    const allValues = data.data.flatMap((row) =>
      data.model_names.map((m) => row[m] as number | null).filter((v): v is number => v !== null)
    );
    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);

    const heatmapData: (number | string)[][] = [];
    const nullData: (number | string)[][] = [];

    data.data.forEach((row, rowIndex) => {
      data.model_names.forEach((model, colIndex) => {
        const val = row[model];
        if (val === null) {
          nullData.push([colIndex, data.row_labels.length - 1 - rowIndex, 'null_marker']);
        } else {
          heatmapData.push([colIndex, data.row_labels.length - 1 - rowIndex, val as number]);
        }
      });
    });

    const option: echarts.EChartsOption = {
      backgroundColor: '#ffffff',
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          if (params.data[2] === 'null_marker') {
            const rowLabel = data.row_labels[data.row_labels.length - 1 - params.data[1]];
            const modelName = data.model_names[params.data[0]];
            return `${rowLabel}<br/>${modelName}: —`;
          }
          const rowLabel = data.row_labels[data.row_labels.length - 1 - params.data[1]];
          const modelName = data.model_names[params.data[0]];
          const value = (params.data[2] as number).toFixed(2);
          return `${rowLabel}<br/>${modelName}: ${value}%`;
        },
      },
      visualMap: {
        min: dataMin,
        max: dataMax,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: isMobile ? 10 : 20,
        textStyle: {
          color: '#4A4A4A',
          fontSize: isMobile ? 10 : 12,
        },
        inRange: {
          color: HEATMAP_COLORS,
        },
      },
      grid: {
        left: isMobile ? 60 : 120,
        right: isMobile ? 20 : 40,
        top: isMobile ? 20 : 40,
        bottom: isMobile ? 70 : 80,
        containLabel: false,
      },
      xAxis: {
        type: 'category',
        data: data.model_names,
        splitArea: { show: true },
        axisLabel: {
          color: '#4A4A4A',
          fontWeight: 'bold',
          fontSize: isMobile ? 9 : 11,
          rotate: isMobile ? 30 : 0,
          interval: 0,
        },
        axisLine: {
          lineStyle: { color: '#333333' },
        },
      },
      yAxis: {
        type: 'category',
        data: [...data.row_labels].reverse(),
        splitArea: { show: true },
        axisLabel: {
          color: '#4A4A4A',
          fontSize: isMobile ? 9 : 11,
        },
        axisLine: {
          lineStyle: { color: '#333333' },
        },
      },
      series: [
        {
          name: 'Performance',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: (params: any) => (params.data[2] as number).toFixed(1) + '%',
            fontSize: isMobile ? 8 : 10,
            fontWeight: 'bold',
            color: '#333333',
          },
          emphasis: {
            itemStyle: {
              borderColor: '#333333',
              borderWidth: 1,
            },
          },
          itemStyle: {
            borderColor: '#ffffff',
            borderWidth: 1,
          },
        },
        {
          name: 'Missing',
          type: 'scatter',
          data: nullData,
          symbol: 'rect',
          symbolSize: 20,
          itemStyle: {
            color: NULL_COLOR,
            borderColor: '#ffffff',
            borderWidth: 1,
          },
          label: {
            show: true,
            formatter: () => '—',
            fontSize: isMobile ? 8 : 10,
            fontWeight: 'bold',
            color: '#888888',
          },
          emphasis: {
            itemStyle: {
              borderColor: '#333333',
              borderWidth: 1,
            },
          },
        },
      ],
      animationDuration: 800,
      animationEasing: 'cubicOut',
    };

    const chart = echarts.getInstanceByDom(container);
    if (chart) {
      chart.dispose();
    }

    const initChart = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w <= 0 || h <= 0) return;
      const newChart = echarts.init(container);
      chartInstanceRef.current = newChart;
      newChart.setOption(option);
    };

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
          initChart();
          observer.disconnect();
          break;
        }
      }
    });
    observer.observe(container);

    initChart();

    const handleResize = () => {
      const c = chartInstanceRef.current;
      if (c) c.resize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', handleResize);
      const c = chartInstanceRef.current;
      if (c) c.dispose();
    };
  }, [data]);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
          fontSize: 16,
          color: '#666',
        }}>
          {t('Loading chart...')}
        </div>
      );
    }

    if (isError) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
          fontSize: 16,
          color: '#d32f2f',
        }}>
          {t('Failed to load chart data: {message}').replace('{message}', '')}
        </div>
      );
    }

    if (!data) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
          fontSize: 16,
          color: '#666',
        }}>
          No data available
        </div>
      );
    }

    return (
      <div
        ref={chartRef}
        style={{
          width: '100%',
          height: window.innerWidth < 768 ? 500 : 600,
        }}
      />
    );
  };

  return (
    <div
      style={{
        backgroundColor: MORANDI.cardBg,
        borderRadius: 12,
        padding: 24,
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
      }}
    >
      {renderContent()}
    </div>
  );
};

export default DatasetComparisonHeatmap;