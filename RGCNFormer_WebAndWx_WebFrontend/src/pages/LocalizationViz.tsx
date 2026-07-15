/**
 * LocalizationViz.tsx - 12 类修饰定位概率曲线 + 饼图 / 12-class localization viz
 *
 * /classification 之一(也可独立访问)。使用 ECharts 渲染两层可视化:
 *   1. 折线图:横轴 = 序列位置,纵轴 = 12 类修饰在该位置的概率(多 line,每类一色)
 *   2. 饼图:全序列上每类修饰的"被预测为正"位置占比
 * 数据由 fetchMrmodnLocalization 拉取,经 useQuery 缓存。莫兰迪配色。
 * One of the /classification pages. Renders two ECharts views:
 *   1. Line chart: x = sequence position, y = per-class probability (one line per mod)
 *   2. Pie chart: per-class share of "predicted positive" positions across the sequence
 * Data via fetchMrmodnLocalization + useQuery. Morandi palette.
 *
 * 功能模块 / Modules:
 * - 多 line 折线图(12 系列)/ 12-series line chart
 * - 饼图(预测占比)/ Pie chart of predicted shares
 * - xAxis 同步缩放(dataZoom)/ Linked dataZoom
 * - tooltip(位置 + 概率)/ Position + probability tooltip
 *
 * 输入 / Inputs:
 * - useRna().rnaSequence: 当前 RNA 序列 / current RNA sequence
 * - 后端 /api/v1/mrmodn-localization 返回 MrmodnLocalizationData
 *
 * 输出 / Outputs:
 * - JSX.Element 折线 + 饼图 / Line + pie JSX
 *
 * 数据流 / Data Flow:
 * 1. useRna().rnaSequence 变化 → 触发 useQuery 重新拉取
 * 2. fetchMrmodnLocalization(seq) → matrix[L, 12] + sequence label
 * 3. 折线图 series.data = matrix
 * 4. 饼图 series.data = matrix.sum(axis=0) 归一化
 * 5. echart.setOption 渲染
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchMrmodnLocalization)、context/RnaContext
 * - 被调用 / Called by: App.tsx(<Route path="/classification">)
 * - 关联 / Related: LocComparisonViz.tsx(多模型对比)、ClassificationViz.tsx
 *
 * 使用示例 / Usage Example:
 *   <Route path="/classification" element={<LocalizationViz />} />
 *   // 浏览器 /mrmodn/classification
 */
import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import type { ECharts } from 'echarts';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { useQuery } from '@tanstack/react-query';
import { fetchMrmodnLocalization, type MrmodnLocalizationData } from '../lib/api';

const MORANDI = {
  bgCircle: '#EBEBEB',
  pieFill: '#84A59D',
  textDark: '#4A4A4A',
  gridLine: '#F0F0F0',
  cardBg: '#FFFFFF',
};

interface LocalizationVizProps {
  data?: MrmodnLocalizationData;
}

const LocalizationViz: React.FC<LocalizationVizProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<ECharts | null>(null);

  const { data: queryData, isLoading, isError } = useQuery({
    queryKey: ['mrmodnLocalization'],
    queryFn: fetchMrmodnLocalization,
    enabled: !propData,
  });

  const data = propData || queryData;

  useEffect(() => {
    if (!chartRef.current || !data) return;

    const container = chartRef.current;
    const isMobile = window.innerWidth < 768;

    const nRows = data.classes.length;
    const donutR = isMobile ? 8 : 11*1.5;
    const donutRGap = donutR * 0.82;

    // Build indicator markers: one per class per statistic type

    const option = {
      backgroundColor: '#ffffff',
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.seriesType === 'custom') {
            const col = params.data[0];
            const row = params.data[1];
            const value = params.data[2];
            const classIdx = nRows - 1 - row;
            const className = data.class_names[classIdx];
            const kLabel = data.k_labels[col];
            return (
              `${className}<br/>` +
              `${kLabel}: ${(value * 100).toFixed(1)}%`
            );
          }
          return '';
        },
      },
      grid: {
        left: isMobile ? 60 : 80,
        right: isMobile ? 20 : 40,
          top: isMobile ? 20 : 30,
        bottom: isMobile ? 70 : 90,
        containLabel: false,
      },
      xAxis: {
        type: 'category',
        data: data.k_labels,
        position: 'top',
        axisLabel: {
          color: MORANDI.textDark,
          fontWeight: 'bold',
          fontSize: isMobile ? 9 : 11*2,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'category',
        data: [...data.class_names].reverse(),
        axisLabel: {
          color: MORANDI.textDark,
          fontWeight: 'bold',
          fontSize: isMobile ? 9 : 11*2,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: {
          show: true,
          lineStyle: { color: MORANDI.gridLine, type: 'dashed', width: 1 },
        },
      },
      series: [
        {
          type: 'custom',
          name: 'Donuts',
          renderItem: (_params: any, api: any) => {
            const col = api.value(0);
            const row = api.value(1);
            const value = api.value(2);
            const cx = api.coord([col, row])[0];
            const cy = api.coord([col, row])[1];

            return {
              type: 'group',
              children: [
                {
                  type: 'sector',
                  shape: { cx, cy, r0: 0, r: donutR, startAngle: 0, endAngle: Math.PI * 2 },
                  style: { fill: MORANDI.bgCircle, stroke: 'none' },
                },
                {
                  type: 'sector',
                  shape: { cx, cy, r0: 0, r: donutRGap, startAngle: -Math.PI / 2, endAngle: -Math.PI / 2 + Math.PI * 2 * value },
                  style: { fill: MORANDI.pieFill, stroke: MORANDI.pieFill, lineWidth: 0.5 },
                },
              ],
            };
          },
          data: data.heatmap.flatMap((row, rowIdx) =>
            row.map((val, colIdx) => [colIdx, nRows - 1 - rowIdx, val])
          ),
          z: 1,
        },
      ],
      graphic: {
        elements: [
          {
            type: 'text',
            left: 12,
            bottom: isMobile ? 52 : 62,
            style: {
              text: t('Reference'),
              fill: MORANDI.textDark,
              fontWeight: 'bold',
              fontSize: isMobile ? 10 : 12,
            },
          },
          ...[1.0, 0.75, 0.5, 0.25].flatMap((val, i) => {
            const refRad = isMobile ? 8 : 11;
            const gap = refRad * 3.5;
            const y = (container.clientHeight || 600) - (isMobile ? 34 : 48);
            const cx = 16 + refRad + i * gap;
            const labels = ['100%', '75%', '50%', '25%'];
            return [
              {
                type: 'circle' as const,
                shape: { cx, cy: y, r: refRad },
                style: { fill: MORANDI.bgCircle, stroke: 'none' },
              },
              {
                type: 'sector' as const,
                shape: { cx, cy: y, r0: 0, r: refRad * 0.82, startAngle: -Math.PI / 2, endAngle: -Math.PI / 2 + Math.PI * 2 * val },
                style: { fill: MORANDI.pieFill, stroke: 'none' },
              },
              {
                type: 'text' as const,
                left: cx,
                top: y + refRad + 8,
                style: {
                  text: labels[i],
                  fill: MORANDI.textDark,
                  fontWeight: 'bold',
                  fontSize: 9,
                  textAlign: 'center',
                },
              },
            ];
          }),
        ],
      },
      animationDuration: 800,
      animationEasing: 'cubicOut' as any,
    };

    const chart = echarts.getInstanceByDom(container);
    if (chart) chart.dispose();

    const initChart = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w <= 0 || h <= 0) return;
      const newChart = echarts.init(container);
      chartInstanceRef.current = newChart;
      newChart.setOption(option as any);
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
  }, [data, t]);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, fontSize: 16, color: '#666' }}>
          {t('Loading chart...')}
        </div>
      );
    }
    if (isError) {
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, fontSize: 16, color: '#d32f2f' }}>
          {t('Failed to load chart data: {message}').replace('{message}', '')}
        </div>
      );
    }
    if (!data) {
      return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, fontSize: 16, color: '#666' }}>
          No data available
        </div>
      );
    }
    return (
      <div ref={chartRef} style={{ width: '100%', height: window.innerWidth < 768 ? 450 : 600 }} />
    );
  };

  return (
    <div style={{ backgroundColor: MORANDI.cardBg, borderRadius: 12, padding: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
      {renderContent()}
    </div>
  );
};

export default LocalizationViz;
