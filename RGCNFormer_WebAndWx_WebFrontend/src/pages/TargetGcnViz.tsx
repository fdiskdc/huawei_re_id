/**
 * TargetGcnViz.tsx - 目标节点 GCN 3D 可视化(节点高亮 + 边聚合)/ Target-node GCN viz
 *
 * /target-gcn 路由页面。在 GcnViz 基础上加入"目标节点"概念:用户可指定一个
 * targetNodeIndex,后端 fetchGcnAggregation 拉回该节点周围的 N 跳邻居边权重
 * 聚合数据,前端以不同颜色/粗细高亮目标节点与连接边,便于分析 GCN 信息传递。
 * Page mounted at /target-gcn. Extends GcnViz with a "target node" concept: the user
 * supplies a targetNodeIndex, the backend returns fetchGcnAggregation data (per-edge
 * aggregated weights around the target), and the frontend highlights the target node
 * and incident edges with distinct colors/widths to analyze GCN message passing.
 *
 * 功能模块 / Modules:
 * - 目标节点索引输入(InputNumber)/ Target node index input
 * - 边聚合数据获取(fetchGcnAggregation)/ Fetch aggregated edge data
 * - 目标节点高亮(不同色)/ Target node highlight
 * - 邻居边粗细/颜色按权重映射 / Neighbor edge width/color by weight
 *
 * 输入 / Inputs:
 * - targetNodeIdx: number / index of the target nucleotide
 * - useRna().rnaSequence: 当前序列 / current sequence
 *
 * 输出 / Outputs:
 * - JSX.Element 3D 图(目标节点突出)/ 3D graph with highlighted target
 *
 * 数据流 / Data Flow:
 * 1. 用户输入 targetNodeIdx → 提交
 * 2. fetchGcnAggregation(seq, idx) → 返回 target 节点 + 邻居边权重
 * 3. 渲染 ForceGraph3D,目标节点大尺寸 + 醒目色
 * 4. 邻居边按 weight 映射到线宽/颜色
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchGcnAggregation)、context/RnaContext
 * - 被调用 / Called by: App.tsx(<Route path="/target-gcn">)
 * - 关联 / Related: GcnViz.tsx(基础 GCN 可视化)
 *
 * 使用示例 / Usage Example:
 *   <Route path="/target-gcn" element={<TargetGcnViz />} />
 *   // 浏览器 /mrmodn/target-gcn
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Spin, Alert, Button, Space, Card, Typography, InputNumber } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useRna } from '../context/RnaContext';
import { useTranslation } from '../lib/i18n/LanguageContext';
import { fetchGcnAggregation } from '../lib/api';

const { Title, Text } = Typography;

interface GraphNode {
  id: string;
  data?: {
    index: number;
    type: string;
  };
  type?: string;
}

interface GraphEdge {
  source: string;
  target: string;
}

interface AggregationResponse {
  targetNode: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  aggregationData: {
    layer: number;
    messages: { from: number; strength: number }[];
  }[];
}

interface TargetGcnVizProps {
  targetNodeIdx?: number;
  data?: AggregationResponse;
}

const TargetGcnViz: React.FC<TargetGcnVizProps> = ({ targetNodeIdx: propTargetNodeIdx, data: propData }) => {
  const { t } = useTranslation();
  const { rnaSequence } = useRna();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aggregationData, setAggregationData] = useState<AggregationResponse | null>(null);
  const [targetNode, setTargetNode] = useState<number | null>(propTargetNodeIdx || null);
  const [inputTargetNodeIdx, setInputTargetNodeIdx] = useState<number | null>(propTargetNodeIdx || null);
  const [hasComputed, setHasComputed] = useState(false);
  
  const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] }>({ nodes: [], links: [] });
  const [hopDistances, setHopDistances] = useState<Map<number, number>>(new Map());
  const [maxHopDistance, setMaxHopDistance] = useState<number>(0);
  const graphRef = useRef<any>(null);
  const neighborNodeIndicesRef = useRef<Set<number>>(new Set());

  // Morandi colors
  const COLORS = {
    background: '#F5F5F5', // Light gray (Morandi)
    defaultNode: '#D8BFD8', // Thistle/gray-pink (Morandi)
    defaultLink: '#D3D3D3', // Light gray (Morandi)
    targetNode: '#FF7F50', // Coral (highlight)
    neighborNode: '#40E0D0', // Turquoise/cyan (highlight)
    targetLink: '#FF7F50', // Coral (matching target node)
    neighborLink: '#40E0D0', // Turquoise (matching neighbor node)
  };

  // Node sizes
  const NODE_SIZES = {
    target: 80,    // Large for target node
    neighbor: 60,   // Medium for neighbor nodes
    normal: 4,     // Small for normal nodes
  };

  const handleCompute = async () => {
    if (!rnaSequence || inputTargetNodeIdx === null) {
      setError(t('Please provide RNA sequence and target node index.'));
      return;
    }

    setLoading(true);
    setError(null);
    setHasComputed(false);

    try {
      const data: AggregationResponse = await fetchGcnAggregation({
        rnaSequence: rnaSequence,
        targetNodeIdx: inputTargetNodeIdx
      });
      console.log('API Response Data:', data);
      setAggregationData(data);
      setTargetNode(data.targetNode);

      const nodes = data.nodes.map(node => ({
        id: node.id,
        ...node.data,
        label: `${node.data?.type}${node.data?.index}`,
      }));

      const links = data.edges.map(edge => ({
        source: edge.source,
        target: edge.target,
      }));

      setGraphData({ nodes, links });
      
      // BFS to find N-hop neighbors
      const nHops = data.aggregationData.length;
      const distances = new Map<number, number>();
      const queue: [number, number][] = [[data.targetNode, 0]];
      const visited = new Set<number>([data.targetNode]);
      distances.set(data.targetNode, 0);

      const adjacencyList = new Map<number, number[]>();
      links.forEach(link => {
        const sourceNode = nodes.find(n => n.id === link.source);
        const targetNodeObj = nodes.find(n => n.id === link.target);
        if (sourceNode && targetNodeObj && sourceNode.index !== undefined && targetNodeObj.index !== undefined) {
          if (!adjacencyList.has(sourceNode.index)) adjacencyList.set(sourceNode.index, []);
          if (!adjacencyList.has(targetNodeObj.index)) adjacencyList.set(targetNodeObj.index, []);
          adjacencyList.get(sourceNode.index)!.push(targetNodeObj.index);
  adjacencyList.get(targetNodeObj.index)!.push(sourceNode.index);
        }
      });

      let head = 0;
      while(head < queue.length) {
        const [currentNode, distance] = queue[head++];
        if (distance >= nHops) continue;

        const neighbors = adjacencyList.get(currentNode) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            distances.set(neighbor, distance + 1);
            queue.push([neighbor, distance + 1]);
          }
        }
      }
      setHopDistances(distances);
      
      // Calculate max hop distance for legend
      let maxDist = 0;
      distances.forEach((dist) => {
        if (dist > maxDist) maxDist = dist;
      });
      setMaxHopDistance(maxDist);
      
      // Create neighborNodeIndices (all nodes with distance > 0)
      const neighborIndices = new Set<number>();
      distances.forEach((dist, nodeIdx) => {
        if (dist > 0) {
          neighborIndices.add(nodeIdx);
        }
      });
      neighborNodeIndicesRef.current = neighborIndices;
      
      setHasComputed(true);

    } catch (e: any) {
      setError(t('Failed to load GCN aggregation data: {message}').replace('{message}', e.message));
    } finally {
      setLoading(false);
    }
  };

  // Process propData when provided (for Modal usage)
  useEffect(() => {
    if (propData) {
      const data = propData;
      setAggregationData(data);
      setTargetNode(data.targetNode);

      const nodes = data.nodes.map(node => ({
        id: node.id,
        ...node.data,
        label: `${node.data?.type}${node.data?.index}`,
      }));

      const links = data.edges.map(edge => ({
        source: edge.source,
        target: edge.target,
      }));

      setGraphData({ nodes, links });

      // BFS to find N-hop neighbors
      const nHops = data.aggregationData.length;
      const distances = new Map<number, number>();
      const queue: [number, number][] = [[data.targetNode, 0]];
      const visited = new Set<number>([data.targetNode]);
      distances.set(data.targetNode, 0);

      const adjacencyList = new Map<number, number[]>();
      links.forEach(link => {
        const sourceNode = nodes.find(n => n.id === link.source);
        const targetNodeObj = nodes.find(n => n.id === link.target);
        if (sourceNode && targetNodeObj && sourceNode.index !== undefined && targetNodeObj.index !== undefined) {
          if (!adjacencyList.has(sourceNode.index)) adjacencyList.set(sourceNode.index, []);
          if (!adjacencyList.has(targetNodeObj.index)) adjacencyList.set(targetNodeObj.index, []);
          adjacencyList.get(sourceNode.index)!.push(targetNodeObj.index);
          adjacencyList.get(targetNodeObj.index)!.push(sourceNode.index);
        }
      });

      let head = 0;
      while (head < queue.length) {
        const [currentNode, distance] = queue[head++];
        if (distance >= nHops) continue;
        const neighbors = adjacencyList.get(currentNode) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            distances.set(neighbor, distance + 1);
            queue.push([neighbor, distance + 1]);
          }
        }
      }
      setHopDistances(distances);

      let maxDist = 0;
      distances.forEach((dist) => {
        if (dist > maxDist) maxDist = dist;
      });
      setMaxHopDistance(maxDist);

      const neighborIndices = new Set<number>();
      distances.forEach((dist, nodeIdx) => {
        if (dist > 0) {
          neighborIndices.add(nodeIdx);
        }
      });
      neighborNodeIndicesRef.current = neighborIndices;

      setHasComputed(true);
    }
  }, [propData]);

  const onNodeClick = useCallback((node: any) => {
    setInputTargetNodeIdx(node.index);
  }, []);

  // Initial camera focus on target node after graph settles
  useEffect(() => {
    if (hasComputed && aggregationData && graphRef.current && graphData.nodes.length > 0) {
      const timer = setTimeout(() => {
        const targetNodeData = graphData.nodes.find(n => n.index === aggregationData.targetNode);
        const centerX = targetNodeData?.x ?? 0;
        const centerY = targetNodeData?.y ?? 0;
        const centerZ = targetNodeData?.z ?? 0;

        graphRef.current?.cameraPosition(
          { x: centerX + 200, y: centerY + 150, z: centerZ + 200 },
          { x: centerX, y: centerY, z: centerZ },
          0
        );
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [hasComputed, aggregationData, graphData]);

  if (!propData && !rnaSequence) {
    return (
      <Alert
        message={t('Error')}
        description={t('Please provide RNA sequence and target node index.')}
        type="error"
        showIcon
      />
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Introduction Card */}
      <Card style={{ marginBottom: 16, background: '#fff2e8', borderColor: '#fa541c' }}>
        <Typography.Title level={4} style={{ color: '#fa541c', margin: 0 }}>{t('GCN Message Passing')}</Typography.Title>
        <Typography.Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
          {t('The GCN Message Passing visualization shows how information propagates through the Graph Convolutional Network layers for a specific target node. Select a node index to see its neighbors and the message flow between them. The target node is highlighted in coral, its neighbors in turquoise, and animated particles show the direction of message passing.')}
        </Typography.Paragraph>
      </Card>

      <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {loading && (
        <Spin
          tip={t('Loading GCN aggregation data...')}
          size="large"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 100,
          }}
        />
      )}

      {error && (
        <Alert
          message={t('Error')}
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          closable
          onClose={() => setError(null)}
        />
      )}

      {!propData && (
      <Card title={t('GCN Message Passing')} style={{ marginBottom: 0 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <Text strong>{t('Target Node Index')}:</Text>
            <InputNumber
              min={0}
              max={rnaSequence ? rnaSequence.length - 1 : 100}
              value={inputTargetNodeIdx}
              onChange={(value) => setInputTargetNodeIdx(value)}
              placeholder={t('Enter target node index')}
              style={{ width: 200 }}
            />
            <Text type="secondary">({t('Range')}: 0 - {rnaSequence ? rnaSequence.length - 1 : '?'})</Text>
            {!hasComputed && (
              <Button
                type="primary"
                icon={<SearchOutlined />}
                onClick={handleCompute}
                loading={loading}
                disabled={inputTargetNodeIdx === null}
              >
                {t('Compute')}
              </Button>
            )}
            {targetNode !== null && (
              <Text type="secondary" style={{ marginLeft: '8px' }}>
                {t('Current selected')} {t('Target Node')}: {targetNode}
              </Text>
            )}
          </div>
          <div>
            <Text type="secondary">
              {t('Enter a node index to visualize message passing from its neighbors through GCN layers')}
            </Text>
          </div>
          <Text type="secondary">
            {t('Tip: You can also click on any node in the graph to select it as the target node.')}
          </Text>
        </Space>
      </Card>
      )}

      {aggregationData && hasComputed && (
        <>
          {/* Graph Visualization - 3D ForceGraph */}
          <div
            style={{
              flex: 1,
              minHeight: '500px',
              width: '100%',
              position: 'relative',
              background: COLORS.background,
              borderRadius: '8px',
              overflow: 'hidden',
              border: '1px solid #d9d9d9',
            }}
          >
            <ForceGraph3D
              ref={graphRef}
              graphData={graphData}
              onNodeClick={onNodeClick}
              nodeLabel="label"
              nodeResolution={16}
              nodeVal={(node) => {
                if (!aggregationData) return NODE_SIZES.normal;

                const nodeIndex = node.index;
                const hopDistance = hopDistances.get(nodeIndex);

                // Target node (distance === 0) - largest
                if (hopDistance === 0) {
                  return NODE_SIZES.target;
                }

                // Neighbor nodes - size decreases with hop distance
                if (hopDistance !== undefined && hopDistance > 0) {
                  // Calculate size: starts at NODE_SIZES.neighbor for distance 1,
                  // decreases to a minimum of NODE_SIZES.normal for larger distances
                  const sizeDecrement = (hopDistance - 1) * 10;
                  const size = Math.max(NODE_SIZES.normal, NODE_SIZES.neighbor - sizeDecrement);
                  return size;
                }

                // Default nodes - smallest
                return NODE_SIZES.normal;
              }}
              nodeColor={(node) => {
                if (!aggregationData) return COLORS.defaultNode;

                const nodeIndex = node.index;
                const hopDistance = hopDistances.get(nodeIndex);

                // Target node (distance === 0) - coral color (opaque)
                if (hopDistance === 0) {
                  return COLORS.targetNode;
                }

                // 1-hop neighbor (distance === 1) - color based on strength
                if (hopDistance === 1) {
                  // Get strength from aggregationData (Layer 0)
                  const layer0Messages = aggregationData.aggregationData[0]?.messages || [];
                  const message = layer0Messages.find((msg) => msg.from === nodeIndex);
                  
                  if (message) {
                    // Normalize strength to alpha range [0.3, 1.0]
                    // Assuming strength values are in [0, 1] range
                    const normalizedStrength = Math.max(0.3, Math.min(1.0, message.strength));
                    return `rgba(64, 224, 208, ${normalizedStrength})`; // Turquoise with alpha
                  }
                  
                  // Default if no strength found
                  return `rgba(64, 224, 208, 0.7)`;
                }

                // 2+ hop neighbors (distance > 1) - opacity based on distance
                if (hopDistance !== undefined && hopDistance > 1) {
                  // Opacity decreases with distance (1 / hopDistance)
                  const alpha = Math.max(0.2, 1.0 / hopDistance);
                  return `rgba(64, 224, 208, ${alpha})`; // Turquoise with decreasing alpha
                }

                // Default nodes - very dim Morandi thistle color
                return 'rgba(216, 191, 216, 0.3)';
              }}
              linkColor={(link) => {
                if (!aggregationData) return COLORS.defaultLink;

                // Find source and target nodes
                const sourceNode = graphData.nodes.find(n => n.id === link.source);
                const targetNodeObj = graphData.nodes.find(n => n.id === link.target);
                
                if (!sourceNode || !targetNodeObj) return COLORS.defaultLink;

                const sourceHopDistance = hopDistances.get(sourceNode.index);
                const targetHopDistance = hopDistances.get(targetNodeObj.index);

                // Get minimum hop distance between the two nodes
                const minHopDistance = sourceHopDistance !== undefined && targetHopDistance !== undefined
                  ? Math.min(sourceHopDistance, targetHopDistance)
                  : undefined;

                // Check if this link connects to the target node
                if (sourceHopDistance === 0 || targetHopDistance === 0) {
                  return COLORS.targetLink;
                }

                // Check if at least one endpoint is a neighbor (has hop distance)
                if (minHopDistance !== undefined && minHopDistance > 0) {
                  // Use turquoise color with opacity based on min hop distance
                  const alpha = Math.max(0.2, 1.0 / minHopDistance);
                  return `rgba(64, 224, 208, ${alpha})`;
                }

                // Default links - light gray Morandi color
                return COLORS.defaultLink;
              }}
              linkWidth={(link) => {
                if (!aggregationData) return 1;

                // Find source and target nodes
                const sourceNode = graphData.nodes.find(n => n.id === link.source);
                const targetNodeObj = graphData.nodes.find(n => n.id === link.target);
                
                if (!sourceNode || !targetNodeObj) return 1;

                const sourceHopDistance = hopDistances.get(sourceNode.index);
                const targetHopDistance = hopDistances.get(targetNodeObj.index);

                // Get minimum hop distance between the two nodes
                const minHopDistance = sourceHopDistance !== undefined && targetHopDistance !== undefined
                  ? Math.min(sourceHopDistance, targetHopDistance)
                  : undefined;

                // Check if this link connects to the target node
                if (sourceHopDistance === 0 || targetHopDistance === 0) {
                  return 15;
                }

                // Scale width based on minimum hop distance (closer = thicker)
                if (minHopDistance !== undefined && minHopDistance > 0) {
                  return Math.max(2, 15 - (minHopDistance - 1) * 3);
                }

                // Default links - thin
                return 1;
              }}
              linkDirectionalParticles={(link) => {
                if (!aggregationData) return 0;

                // Find source and target nodes
                const sourceNode = graphData.nodes.find(n => n.id === link.source);
                const targetNodeObj = graphData.nodes.find(n => n.id === link.target);
                
                if (!sourceNode || !targetNodeObj) return 0;

                const sourceHopDistance = hopDistances.get(sourceNode.index);
                const targetHopDistance = hopDistances.get(targetNodeObj.index);

                // Get minimum hop distance between the two nodes
                const minHopDistance = sourceHopDistance !== undefined && targetHopDistance !== undefined
                  ? Math.min(sourceHopDistance, targetHopDistance)
                  : undefined;

                // Animate links connected to target node - more particles
                if (sourceHopDistance === 0 || targetHopDistance === 0) {
                  return 4;
                }

                // Scale particle count based on minimum hop distance (closer = more particles)
                if (minHopDistance !== undefined && minHopDistance > 0) {
                  return Math.max(1, 4 - (minHopDistance - 1));
                }

                // Default links - no animation
                return 0;
              }}
              linkDirectionalParticleWidth={2}
              linkDirectionalParticleSpeed={0.008}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={0.85}
              enableNodeDrag={true}
              backgroundColor={COLORS.background}
              width={undefined}
              height={undefined}
              onEngineStop={() => {
                if (graphRef.current) {
                  graphRef.current.zoomToFit(400);
                }
              }}
            />
          </div>

          {/* Dynamic Legend */}
          <Card size="small" style={{ padding: '12px' }}>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Title level={5} style={{ margin: 0 }}>{t('Legend')}</Title>
              
              {/* Target Node */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: COLORS.targetNode }}></div>
                <Text>{t('Target Node')}</Text>
              </div>

              {/* Dynamic Hop Distance Legend */}
              {maxHopDistance > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <Text strong style={{ fontSize: '13px' }}>{t('Hop Neighbors')}:</Text>
                  
                  {/* 1-Hop Neighbors - with strength variation */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <Text style={{ fontSize: '12px' }}>{t('1-Hop')}:</Text>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'rgba(64, 224, 208, 1.0)' }}></div>
                      <Text style={{ fontSize: '11px', color: '#666' }}>{t('High Strength')}</Text>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'rgba(64, 224, 208, 0.5)' }}></div>
                      <Text style={{ fontSize: '11px', color: '#666' }}>{t('Medium Strength')}</Text>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'rgba(64, 224, 208, 0.3)' }}></div>
                      <Text style={{ fontSize: '11px', color: '#666' }}>{t('Low Strength')}</Text>
                    </div>
                  </div>

                  {/* 2+ Hop Neighbors */}
                  {maxHopDistance >= 2 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <Text style={{ fontSize: '12px' }}>{t('2+ Hop')}:</Text>
                      {Array.from({ length: Math.min(3, maxHopDistance - 1) }, (_, i) => {
                        const hop = i + 2;
                        const alpha = Math.max(0.2, 1.0 / hop);
                        return (
                          <div key={hop} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: `rgba(64, 224, 208, ${alpha.toFixed(2)})` }}></div>
                            <Text style={{ fontSize: '11px', color: '#666' }}>{hop}-{t('Hop')}</Text>
                          </div>
                        );
                      })}
                      {maxHopDistance > 4 && (
                        <Text style={{ fontSize: '11px', color: '#666' }}>({t('...')})</Text>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Normal Nodes */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: COLORS.defaultNode }}></div>
                <Text>{t('Normal Node')}</Text>
              </div>

              {/* Links Legend */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '8px' }}>
                <Text style={{ fontSize: '12px' }}>{t('Links')}:</Text>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: '40px', height: '3px', backgroundColor: COLORS.targetLink }}></div>
                    <Text style={{ fontSize: '12px' }}>{t('Target')}</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: '40px', height: '2px', backgroundColor: 'rgba(64, 224, 208, 1.0)' }}></div>
                    <Text style={{ fontSize: '12px' }}>{t('1-Hop')}</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: '40px', height: '1px', backgroundColor: 'rgba(64, 224, 208, 0.5)' }}></div>
                    <Text style={{ fontSize: '12px' }}>{t('2+ Hop')}</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <div style={{ width: '40px', height: '1px', backgroundColor: COLORS.defaultLink }}></div>
                    <Text style={{ fontSize: '12px' }}>{t('Normal')}</Text>
                  </div>
                </div>
              </div>

              {/* Note about color/opacity meaning */}
              <div style={{ marginTop: '8px', padding: '8px', background: '#f5f5f5', borderRadius: '4px' }}>
                <Text style={{ fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
                  {t('Color intensity/opacity represents message passing strength or distance to target node.')}
                </Text>
              </div>

              {/* Interaction hint */}
              <div style={{ marginTop: '4px' }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {t('Drag to rotate • Scroll to zoom • Right-click drag to pan • Click node to select as target')}
                </Text>
              </div>
            </Space>
          </Card>
        </>
      )}
      </div>
    </div>
  );
};

export default TargetGcnViz;
