/**
 * MrmodnHeatmap.tsx - mRModN 在某数据集上的分类热力图 / mRModN classification heatmap
 *
 * /viz-display 之一。使用 ECharts 渲染一个 12 × 12(或 12 × N)热力图,行 = 真实
 * 修饰类型,列 = 预测修饰类型,颜色 = 归一化频次或概率。数据由 fetchMrmodnHeatmap
 * 拉取,经 useQuery 缓存。配色与 VizLayout 的莫兰迪色卡一致。
 * One of the /viz-display pages. Renders an ECharts heatmap of size 12×12 (or 12×N):
 * rows = true modification type, columns = predicted modification type, color =
 * normalized frequency/probability. Data via fetchMrmodnHeatmap + useQuery.
 * Colors follow the Morandi palette shared with VizLayout.
 *
 * 功能模块 / Modules:
 * - ECharts 热力图(visualMap 渐变)/ ECharts heatmap with visualMap
 * - 行/列 tooltip(显示计数 + 百分比)/ Tooltip with count + percent
 * - 莫兰迪色卡 / Morandi palette
 * - 加载/错误态 / Loading/error states
 *
 * 输入 / Inputs:
 * - 后端 /api/v1/mrmodn-heatmap 返回 MrmodnHeatmapData
 *
 * 输出 / Outputs:
 * - JSX.Element 热力图 / Heatmap JSX
 *
 * 数据流 / Data Flow:
 * 1. useQuery → fetchMrmodnHeatmap() → matrix + labels
 * 2. ECharts option({ xAxis, yAxis, series: heatmap, visualMap })
 * 3. echart.setOption(option)
 * 4. 鼠标 hover → tooltip 弹出
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchMrmodnHeatmap, MrmodnHeatmapData)
 * - 被调用 / Called by: App.tsx(<Route path="/viz-display"> 内部)
 * - 关联 / Related: DatasetComparisonHeatmap.tsx(跨数据集版本)
 *
 * 使用示例 / Usage Example:
 *   const { data } = useQuery({
 *     queryKey: ['mrmodn-heatmap'],
 *     queryFn: () => fetchMrmodnHeatmap(),
 *   });
 */
import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import type { ECharts } from 'echarts';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { useQuery } from '@tanstack/react-query';
import { fetchMrmodnHeatmap, type MrmodnHeatmapData } from '../lib/api';

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

interface MrmodnHeatmapProps {
  data?: MrmodnHeatmapData;
}

const MrmodnHeatmap: React.FC<MrmodnHeatmapProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<ECharts | null>(null);

  const { data: queryData, isLoading, isError } = useQuery({
    queryKey: ['mrmodnHeatmap'],
    queryFn: fetchMrmodnHeatmap,
    enabled: !propData,
  });

  const data = propData || queryData;

  useEffect(() => {
    if (!chartRef.current || !data) return;

    const container = chartRef.current;
    const isMobile = window.innerWidth < 768;

    const allValues = data.data.flatMap((row) =>
      data.metric_names.map((m) => row[m] as number)
    );
    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);

    const option: echarts.EChartsOption = {
      backgroundColor: '#ffffff',
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const className = data.classes[params.data[1]];
          const metricName = data.metric_names[params.data[0]];
          const value = (params.data[2] as number * 100).toFixed(1);
          return `${className}<br/>${metricName}: ${value}%`;
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
        left: isMobile ? 60 : 100,
        right: isMobile ? 20 : 40,
        top: isMobile ? 20 : 40,
        bottom: isMobile ? 70 : 80,
        containLabel: false,
      },
      xAxis: {
        type: 'category',
        data: data.metric_names,
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
        data: [...data.classes].reverse(),
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
          data: data.data.flatMap((row, rowIndex) =>
            data.metric_names.map((metric, colIndex) => [
              colIndex,
              data.classes.length - 1 - rowIndex,
              row[metric] as number,
            ])
          ),
          label: {
            show: true,
            formatter: (params: any) => ((params.data[2] as number) * 100).toFixed(0) + '%',
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

    // Try to init immediately in case container already has dimensions
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

export default MrmodnHeatmap;