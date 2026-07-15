import React, { useState, useEffect, useMemo } from 'react';
import { Spin, Alert, InputNumber, Button, Space, Card, Select, Typography } from 'antd';
import { ArrowLeftOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { predict } from '../lib/api';

// 12类修饰名称映射 (模型索引 -> 修饰名称)
// 与 backend/human.py 中的 MOD_TO_INDEX 一致 (mod_index - 1 转换后)
const MOD_NAMES = [
  'Am', 'Atol', 'Cm',    // 索引 0-2
  'Gm', 'Tm', 'Y',      // 索引 3-5
  'ac4C', 'm1A', 'm5C', // 索引 6-8
  'm6A', 'm6Am', 'm7G'  // 索引 9-11
];

// 反向映射：模型索引 -> 核苷酸组
// 根据 backend/human.py 中的核苷酸分组转换
const INDEX_TO_NUCLEOTIDE: Record<number, string> = {
  0: 'A', 1: 'A',     // Am, Atol
  2: 'C',             // Cm
  3: 'G',             // Gm
  4: 'U', 5: 'U',     // Tm, Y
  6: 'C',             // ac4C
  7: 'A',             // m1A
  8: 'C',             // m5C
  9: 'A', 10: 'A',    // m6A, m6Am
  11: 'G'             // m7G
};

// 定义权重和API数据的类型
interface Weight {
  index: number;
  type: string;
  score: number;
  originalScore?: number;  // 保存原始分数
  normalizedScore?: number;  // 保存归一化分数
}
interface AttentionData {
  sequence: string;
  weights: Weight[];
}

// 定义莫兰迪配色方案
const baseColors: { [key: string]: string } = {
  'A': '#bcaaa4', // 柔和的灰玫瑰色 (腺嘌呤)
  'G': '#a5d6a7', // 柔和的鼠尾草绿 (鸟嘌呤)
  'C': '#90caf9', // 柔和的石板蓝 (胞嘧啶)
  'U': '#ffe082', // 柔和的沙黄色 (尿嘧啶)
  '-': '#eeeeee', // 用于填充字符的中性灰色
};

// 定义视图窗口的宽度（奇数以保证完美居中）
const VIEWPORT_WIDTH = 101;

interface AttentionVizProps {
  data?: {
    sequence: string;
    weights: Weight[];
  };
}

const AttentionViz: React.FC<AttentionVizProps> = ({ data: propData }) => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(!propData);
  const [error, setError] = useState<string | null>(null);
  // 新增：存储从后端获取的完整数据
  const [sequence, setSequence] = useState('');
  const [allWeights, setAllWeights] = useState<Weight[]>([]);

  const [topX, setTopX] = useState<number>(3);
  const [currentIndex, setCurrentIndex] = useState(0);
  
  // 选中的修饰类型：可以是 'all' 或者修饰类型的name
  const [selectedModificationType, setSelectedModificationType] = useState<string>('all');

  const viewportRef = React.useRef<HTMLDivElement>(null);

  const { rnaSequence, classificationResults } = useRna();

  // 核心逻辑1：获取并存储完整数据
  useEffect(() => {
    // If data is provided via props, use it directly
    if (propData) {
      setSequence(propData.sequence);
      setAllWeights(propData.weights);
      setLoading(false);
      setError(null);
      return;
    }

    // Otherwise, fetch from API
    if (!rnaSequence) {
      setLoading(false);
      setError("No RNA sequence provided.");
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const apiData = await predict({ sequence: rnaSequence });
        const data: AttentionData = apiData.attention;

        console.log('API Response:', apiData);
        console.log('Attention Data:', data);
        console.log('Weights:', data.weights);

        setSequence(data.sequence);
        setAllWeights(data.weights); // 存储所有权重

      } catch (e: any) {
        console.error('Error fetching data:', e);
        setError(t('Failed to load data: {message}').replace('{message}', e.message));
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [rnaSequence, propData]);

  // 根据classificationResults生成Select选项（不包含'all'）
  const selectOptions = useMemo(() => {
    const options: Array<{ value: string; label: string }> = [];
    
    if (classificationResults && classificationResults.length > 0) {
      classificationResults.forEach((result) => {
        // 找到修饰类型在MOD_NAMES中的索引
        const modIndex = MOD_NAMES.indexOf(result.name);
        if (modIndex !== -1) {
          options.push({
            value: result.name,
            label: `${result.name} (${INDEX_TO_NUCLEOTIDE[modIndex]})`
          });
        }
      });
    }
    
    return options;
  }, [classificationResults]);

  // 核心逻辑2：根据 topX 和选中的修饰类型筛选要显示的权重，并进行归一化 (使用 useMemo 优化性能)
  const displayWeights = useMemo(() => {
    if (!allWeights || allWeights.length === 0) return [];

    console.log('Computing displayWeights for modification type:', selectedModificationType);
    console.log('All weights length:', allWeights.length);

    // Step 1: 处理修饰类型选择和剪枝逻辑
    let processedWeights = [...allWeights];

    if (selectedModificationType !== 'all') {
      // 根据选中的修饰类型名称找到对应的索引
      const modIndex = MOD_NAMES.indexOf(selectedModificationType);
      console.log('Modification index:', modIndex);
      
      if (modIndex !== -1) {
        const targetNucleotide = INDEX_TO_NUCLEOTIDE[modIndex];
        console.log('Target nucleotide:', targetNucleotide);
        
        // 打印所有weights的详细信息
        console.log('All weights details:', allWeights.map(w => ({
          type: w.type,
          index: w.index,
          base: sequence[w.index],
          score: w.score
        })));
        
        // 剪枝逻辑：将不匹配目标碱基的权重分数置为0
        // 注意：weight.type字段存储的是核苷酸类型（A/C/G/U），不是修饰类型
        processedWeights = allWeights.map(weight => {
          const positionBase = sequence[weight.index];
          // 处理 T 和 U 的等价性
          const nucleotideMatch = positionBase === targetNucleotide || 
                                   (targetNucleotide === 'U' && positionBase === 'T') ||
                                   (targetNucleotide === 'T' && positionBase === 'U');
          
          console.log(`Weight - Type: ${weight.type}, Index: ${weight.index}, Base: ${positionBase}, TargetBase: ${targetNucleotide}, NucleotideMatch: ${nucleotideMatch}`);
          
          if (!nucleotideMatch) {
            return {
              ...weight,
              score: 0  // 将不匹配的权重分数置为0
            };
          }
          return weight;
        });
        
        console.log('Processed weights after pruning:', processedWeights.filter(w => w.score > 0).length, 'non-zero weights');
        console.log('Non-zero weight types:', processedWeights.filter(w => w.score > 0).map(w => w.type));
      }
    }

    // 关键修复：在归一化之前先过滤掉score为0的权重
    processedWeights = processedWeights.filter(w => w.score > 0);
    console.log('Weights after filtering zeros:', processedWeights.length);

    // 如果没有匹配的权重，直接返回空数组
    if (processedWeights.length === 0) {
      console.log('No matching weights found for modification type:', selectedModificationType);
      return [];
    }

    // Step 2: 按核苷酸组（type）分组
    const groups: { [key: string]: Weight[] } = {};
    processedWeights.forEach(weight => {
      if (!groups[weight.type]) {
        groups[weight.type] = [];
      }
      groups[weight.type].push(weight);
    });

    // Step 3: 对每个组内的分数进行归一化
    const normalizedWeights: Weight[] = [];
    Object.keys(groups).forEach(type => {
      const groupWeights = groups[type];

      // 找到该组内的最大和最小分数
      const scores = groupWeights.map(w => w.score);
      const maxScore = Math.max(...scores);
      const minScore = Math.min(...scores);
      const scoreRange = maxScore - minScore;

      // 归一化到 [0, 1] 范围
      // 如果所有分数相同（range = 0），则都设为 1
      groupWeights.forEach(weight => {
        let normalizedScore: number;
        if (scoreRange === 0) {
          normalizedScore = 1.0;
        } else {
          normalizedScore = (weight.score - minScore) / scoreRange;
        }

        normalizedWeights.push({
          ...weight,
          originalScore: weight.score,  // 保存原始分数
          normalizedScore: normalizedScore,  // 保存归一化分数
          score: normalizedScore  // 使用归一化分数用于排序和显示
        });
      });
    });

    // Step 4: 按归一化后的分数排序并筛选 topX
    return normalizedWeights
      .sort((a, b) => b.score - a.score)
      .slice(0, topX);
  }, [allWeights, topX, selectedModificationType, sequence, classificationResults]);

  // 当 displayWeights 发生显著变化时，重置当前查看的索引
  // 使用 ref 来跟踪上一次的 displayWeights 长度和 selectedModificationType
  const prevDisplayWeightsLength = React.useRef(0);
  const prevSelectedModificationType = React.useRef('all');
  useEffect(() => {
    // 只有当过滤结果数量变化，且不是仅仅因为修改类型改变导致的情况下才重置索引
    // 如果修改类型改变了，但结果数量相同，不重置索引
    // 如果结果数量改变了，才重置索引
    const lengthChanged = displayWeights.length !== prevDisplayWeightsLength.current;
    
    if (lengthChanged) {
      setCurrentIndex(0);
      prevDisplayWeightsLength.current = displayWeights.length;
    }
    
    prevSelectedModificationType.current = selectedModificationType;
  }, [displayWeights, selectedModificationType]);

  // 当 currentIndex 或 displayWeights 变化时，滚动到高亮元素
  // 使用 setTimeout 确保 DOM 已经更新
  useEffect(() => {
    if (viewportRef.current && displayWeights.length > 0) {
      // 使用 setTimeout 确保 DOM 已经完全渲染
      setTimeout(() => {
        const highlightedElement = viewportRef.current?.querySelector('.highlight-container-block');
        if (highlightedElement) {
          highlightedElement.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'center'
          });
        }
      }, 100);
    }
  }, [currentIndex, displayWeights]);

  // 当修饰类型改变时，强制滚动到新的高亮位置
  useEffect(() => {
    if (viewportRef.current && displayWeights.length > 0) {
      // 使用更长的延迟确保 DOM 完全更新
      setTimeout(() => {
        const highlightedElement = viewportRef.current?.querySelector('.highlight-container-block');
        if (highlightedElement) {
          highlightedElement.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'center'
          });
        }
      }, 150);
    }
  }, [selectedModificationType]);

  const handlePrev = () => setCurrentIndex(i => (i > 0 ? i - 1 : i));
  const handleNext = () => setCurrentIndex(i => (i < displayWeights.length - 1 ? i + 1 : i));

  // 处理修饰类型选择变化，添加日志记录
  const handleModificationTypeChange = (value: string) => {
    if (value === 'all') {
      console.log('='.repeat(60));
      console.log('Select Modification Type: All (all nucleotides)');
      console.log('='.repeat(60));
    } else {
      const modIndex = MOD_NAMES.indexOf(value);
      if (modIndex !== -1) {
        const nucleotide = INDEX_TO_NUCLEOTIDE[modIndex];
        console.log('='.repeat(60));
        console.log('Select Modification Type:', value);
        console.log('Corresponding Nucleotide:', nucleotide);
        console.log('='.repeat(60));
      } else {
        console.log('='.repeat(60));
        console.log('Select Modification Type:', value, '(unknown modification)');
        console.log('='.repeat(60));
      }
    }
    setSelectedModificationType(value);
  };

  // 核心逻辑3：生成居中且带填充的序列视图 - 使用彩色方块
  const renderSequenceViewport = () => {
    if (displayWeights.length === 0) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
          <p>{t('No significant modification sites detected.')}</p>
          <p style={{ fontSize: '14px', marginTop: '10px' }}>
            {t('Current sequence length: {length} | Top setting: {topX} | Total weight data: {totalWeights}')
              .replace('{length}', String(sequence.length))
              .replace('{topX}', String(topX))
              .replace('{totalWeights}', String(allWeights.length))}
          </p>
        </div>
      );
    }

    console.log('Rendering viewport with displayWeights:', displayWeights);
    console.log('Current index:', currentIndex);
    console.log('Current highlight:', displayWeights[currentIndex]);

    const currentHighlight = displayWeights[currentIndex];
    const centerIndex = currentHighlight.index;

    const halfWidth = Math.floor(VIEWPORT_WIDTH / 2);
    const startIndex = centerIndex - halfWidth;
    const endIndex = centerIndex + halfWidth;

    console.log('Viewport range:', { startIndex, endIndex, centerIndex, sequenceLength: sequence.length });

    const viewportElements = [];

    for (let i = startIndex; i <= endIndex; i++) {
      const isHighlighted = i === centerIndex;
      const isOutOfBounds = i < 0 || i >= sequence.length;

      let base = isOutOfBounds ? '-' : sequence[i];
      // 在RNA序列中，将T转换为U以保持一致性
      if (base === 'T') {
        base = 'U';
      }
      const bgColor = baseColors[base] || baseColors['-'];

      if (isHighlighted) {
        console.log('Rendering highlighted element at index:', i, 'base:', base);
        viewportElements.push(
          <div key={i} className="highlight-container-block">
            <div className="annotation-label-block">
              {currentHighlight.type} (
                {/* 归一化: {currentHighlight.score.toFixed(3)} | */}
                 {t('Original: {score}').replace('{score}', currentHighlight.originalScore?.toFixed(6) ?? 'N/A')}) ({t('Index: {index}').replace('{index}', String(currentHighlight.index))})
            </div>
            <div
              className="sequence-block highlighted-block"
              style={{ backgroundColor: bgColor }}
            >
              {base}
            </div>
          </div>
        );
      } else {
        viewportElements.push(
          <div
            key={i}
            className="sequence-block"
            style={{ backgroundColor: bgColor }}
          >
            {base}
          </div>
        );
      }
    }
    return viewportElements;
  };

  if (!rnaSequence) {
    return (
      <Alert
        message={t('Error')}
        description={
          <>
            {t('Please enter an RNA sequence.')} <Link to="/">{t('Return to Home')}</Link>
          </>
        }
        type="error"
        showIcon
      />
    );
  }

  if (loading) return <Spin tip={t('Loading sequence data...')} size="large" style={{ display: 'block', marginTop: '50px' }} />;
  if (error) return <Alert message={t('Error')} description={error} type="error" showIcon />;

  // 如果没有修饰类型可选，显示提示信息
  if (selectOptions.length === 0) {
    return (
      <Alert
        message={t('No Modifications Detected')}
        description={t('The model detected no modifications in this RNA sequence.')}
        type="info"
        showIcon
      />
    );
  }

  return (
    <div className="space-wrapper">
      {/* Introduction Card */}
      <Card style={{ marginBottom: 16, background: '#fff7e6', borderColor: '#fa8c16' }}>
        <Typography.Title level={4} style={{ color: '#fa8c16', margin: 0 }}>{t('Attention')}</Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
          {t('The Attention visualization highlights the most important positions in your RNA sequence based on the model\'s attention mechanism. It shows which nucleotides receive the highest attention weights for different RNA modifications. Use the controls to filter by modification type and navigate through the top attention sites to understand the model\'s focus.')}
        </Typography.Paragraph>
      </Card>

      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* 控制区 */}
        <Card title={t('Visualization Controls')}>
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* 修饰类型选择下拉菜单 */}
            <div style={{ marginBottom: '16px' }}>
              <strong>{t('Select Modification Type:')}</strong>
              <div style={{ marginTop: '8px' }}>
                <Select
                  value={selectedModificationType}
                  onChange={handleModificationTypeChange}
                  style={{ width: '100%', maxWidth: '300px' }}
                  options={selectOptions}
                />
              </div>
            </div>
            
            <Space wrap>
              <span>{t('Show Top')}</span>
              <InputNumber min={1} max={1001} value={topX} onChange={(value) => setTopX(value || 1)} />
              <span>{t('modification sites')}</span>
              <Button icon={<ArrowLeftOutlined />} onClick={handlePrev} disabled={currentIndex === 0}>
                {t('Previous')}
              </Button>
              <span>
                {t('Viewing: {current} / {total}').replace('{current}', String(displayWeights.length > 0 ? currentIndex + 1 : 0)).replace('{total}', String(displayWeights.length))}
              </span>
              <Button icon={<ArrowRightOutlined />} onClick={handleNext} disabled={currentIndex >= displayWeights.length - 1}>
                {t('Next')}
              </Button>
            </Space>
          </Space>
        </Card>

        <Card className="card-wrapper">
          {/* 调试信息 */}
          <div style={{ marginBottom: '10px', padding: '10px', background: '#f0f0f0', borderRadius: '4px' }}>
            <strong>{t('Render Debug Info:')}</strong>
            <p>{t('Sequence Length: ')}{sequence.length}</p>
            <p>{t('Current Highlight: ')}{displayWeights.length > 0 ? `${t('Position {index}, Type {type}')
              .replace('{index}', String(displayWeights[currentIndex].index))
              .replace('{type}', displayWeights[currentIndex].type)}` : '无'}</p>
            <p>{t('Viewport Element Count: ')}{displayWeights.length > 0 ? VIEWPORT_WIDTH : 0}</p>
          </div>

          {/* 序列视图 - 彩色方块布局 */}
          <div ref={viewportRef} className="sequence-viewport-container-block" key={`viewport-${selectedModificationType}`}>
            {renderSequenceViewport()}
          </div>
        </Card>

        {/* 显示统计信息 */}
        <Card title={t('Modification Site Statistics')} style={{ marginTop: '20px' }}>
          <p>{t('Original Sequence Length: ')}<strong>{sequence.length}</strong></p>
          <p>{t('Total detected weight data: ')}<strong>{allWeights.length}</strong></p>
          {selectedModificationType !== 'all' && (
            <>
              <p>{t('Selected modification type: ')}<strong>{selectedModificationType}</strong></p>
              {(() => {
                const modIndex = MOD_NAMES.indexOf(selectedModificationType);
                return modIndex !== -1 ? (
                  <p>{t('Target nucleotide: ')}<strong>{INDEX_TO_NUCLEOTIDE[modIndex]}</strong></p>
                ) : null;
              })()}
            </>
          )}
          <p>{t('Currently showing Top: ')}<strong>{topX}</strong></p>
          <p>{t('Actually showing: ')}<strong>{displayWeights.length}</strong> {t(' modification sites')}</p>

          {displayWeights.length > 0 && (
            <div style={{ marginTop: '15px' }}>
              <strong>{t('List of currently displayed modification sites (normalized by nucleotide group):')}</strong>
              <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
                {displayWeights.map((weight, idx) => (
                  <li key={idx}>
                    <strong>{idx + 1}. {weight.type}</strong> - {t('Position: {index},').replace('{index}', String(weight.index))}
                    {/* {t('Normal score: {score}', { score: weight.score.toFixed(4) })} | */}
                    {t('Raw score: {score}').replace('{score}', weight.originalScore?.toFixed(6) ?? 'N/A')}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </Space>
    </div>
  );
};

export default AttentionViz;
