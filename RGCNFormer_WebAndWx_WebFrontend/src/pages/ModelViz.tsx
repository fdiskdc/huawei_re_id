/**
 * ModelViz.tsx - mRModN 模型架构图(react-flow + dagre)/ Model architecture diagram
 *
 * /model-viz 路由页面。使用 react-flow + dagre 自动布局渲染 mRModN 的
 * 模型计算图(节点 = 各层 Conv1d / GCNBlock / Transformer / ClassQueryHead,
 * 边 = 数据流方向)。节点和边数据由 fetchModelGraph 从后端拉取(后端从 ONNX 模型
 * 解析得到 graph JSON)。提供 MiniMap 缩略图、Controls 缩放/平移、连线高亮。
 * Page mounted at /model-viz. Uses react-flow + dagre auto-layout to render the
 * mRModN compute graph (nodes = Conv1d/GCNBlock/Transformer/ClassQueryHead,
 * edges = data flow). Nodes/edges are fetched via fetchModelGraph (backend parses
 * the ONNX model into a graph JSON). Provides MiniMap, zoom/pan Controls, edge highlight.
 *
 * 功能模块 / Modules:
 * - dagre 自动布局(自顶向下)/ dagre auto-layout (top-down)
 * - react-flow 节点/边渲染 / react-flow node/edge rendering
 * - MiniMap + Controls(缩放/平移)/ MiniMap + Controls
 * - 自定义节点类型(可按层着色)/ Custom node types (per-layer color)
 *
 * 输入 / Inputs:
 * - 后端 /api/v1/model-graph 返回 { nodes, edges } / Backend returns { nodes, edges }
 * - dagre 布局参数(ranksep, nodesep, rankdir)/ dagre layout params
 *
 * 输出 / Outputs:
 * - JSX.Element react-flow 画布 / react-flow canvas JSX
 *
 * 数据流 / Data Flow:
 * 1. useEffect → fetchModelGraph() → 拿 graph JSON
 * 2. dagre.layout(g) → 给每个节点计算 x/y
 * 3. 转换 { nodes, edges } 为 react-flow 格式
 * 4. setNodes / setEdges → react-flow 渲染
 * 5. 用户拖拽 / 缩放 / 选中节点
 *
 * 相关文件 / Related Files:
 * - 调用 / Calls: lib/api.ts(fetchModelGraph)、dagre、react-flow
 * - 被调用 / Called by: App.tsx(<Route path="/model-viz">)
 * - 关联 / Related: Cluster_WebAndWx_backend/onnx.py(graph JSON 导出)
 *
 * 使用示例 / Usage Example:
 *   <Route path="/model-viz" element={<ModelViz />} />
 *   // 浏览器 /mrmodn/model-viz
 */
import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
  Position,
  ReactFlowProvider,
} from 'reactflow';
import type { Node, Edge, NodeTypes, Connection } from 'reactflow';
import 'reactflow/dist/style.css';
import { Spin, Alert, Card, Typography, Tooltip } from 'antd';
import { useTranslation } from '../lib/i18n/LanguageContext';
import dagre from 'dagre'; // Import dagre for graph layout
import { fetchModelGraph } from '../lib/api';

const { Title, Text } = Typography;

// --- Dagre Layout Configuration ---
const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

// These can be adjusted based on desired node sizes and spacing
const nodeWidth = 180;
const nodeHeight = 50;

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({ rankdir: direction, ranksep: 50, nodesep: 30 }); // Added ranksep and nodesep for better spacing

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? Position.Left : Position.Top;
    node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom;

    // We are shifting the dagre node position (anchor point is left top) to the center for react flow nodes.
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
    return node;
  });

  return { nodes, edges };
};

// --- Custom Node Component for Graph Operations ---
interface GraphNodeData {
  label: string; // ONNX op_type
  id: string; // ONNX node name
  attributes?: { [key: string]: string };
}

const CustomGraphNode: React.FC<{ data: GraphNodeData }> = ({ data }) => {
  return (
    <Card
      size="small"
      title={<Text strong>{data.label}</Text>}
      style={{ width: nodeWidth, height: nodeHeight, backgroundColor: '#e6f7ff', borderColor: '#91d5ff' }}
      bodyStyle={{ padding: '4px 8px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
    >
      <Tooltip title={data.id}>
        <Text type="secondary" style={{ fontSize: '12px', display: 'block' }}>
          ID: {data.id}
        </Text>
      </Tooltip>
      {data.attributes && Object.keys(data.attributes).length > 0 && (
        <Tooltip
          title={
            <ul>
              {Object.entries(data.attributes).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
          }
        >
          <Text style={{ fontSize: '11px', display: 'block', fontStyle: 'italic' }}>
            (attrs)
          </Text>
        </Tooltip>
      )}
    </Card>
  );
};

const nodeTypes: NodeTypes = {
  graphNode: CustomGraphNode,
};

// --- Main ModelViz Component ---
const ModelViz: React.FC = () => {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Fetch model graph from backend
  useEffect(() => {
    const fetchModelGraphData = async () => {
      try {
        setLoading(true);
        const graphData = await fetchModelGraph();

        // Transform backend JSON to React Flow nodes and edges
        const flowNodes: Node<GraphNodeData>[] = graphData.nodes.map((node: any) => ({
          id: node.id,
          type: 'graphNode',
          position: { x: 0, y: 0 }, // Position will be set by Dagre
          data: {
            label: node.label,
            id: node.id,
            attributes: node.attributes,
          },
        }));

        const flowEdges: Edge[] = graphData.edges.map((edge: any) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          animated: true,
          type: 'smoothstep', // or 'default', 'straight'
          markerEnd: { type: MarkerType.ArrowClosed, color: '#1890ff' },
          style: { stroke: '#1890ff', strokeWidth:1.5 },
        }));

        // Apply Dagre layout
        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
          flowNodes,
          flowEdges,
          'TB' // Top-Bottom layout
        );

        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
        setLoading(false);
      } catch (e: any) {
        console.error('Failed to fetch model graph:', e);
        setError(t('Failed to load model graph: {message}').replace('{message}', e.message));
        setLoading(false);
      }
    };

    fetchModelGraphData();
  }, [t, setNodes, setEdges]);

// Handle new connections (optional for static graph, but useful for interactive features)
const onConnect = useCallback(
  (params: Connection) => setEdges((eds) => addEdge(params, eds)),
  [setEdges]
);

if (loading) {
  return (
    <div style={{ padding: '24px', textAlign: 'center' }}>
      <Spin tip={t('Loading model graph...')} size="large" />
    </div>
  );
}

if (error) {
  return (
    <Alert
      message={t('Error')}
      description={error}
      type="error"
      showIcon
      style={{ margin: '24px' }}
    />
  );
}

return (
  <div style={{ width: '100%', height: 'calc(100vh - 64px)' }}> {/* Adjust height as needed */}
    <div style={{ padding: '16px 24px', background: '#fff', borderBottom: '1px solid #f0f0f0' }}>
      <Title level={4} style={{ margin: 0 }}>
        {t('Model Graph Visualization')}
      </Title>
      <Text type="secondary">
        {t('Interactive data flow visualization of RNA_ClassQuery_Model')}
      </Text>
    </div>
    <ReactFlowProvider> {/* Wrap ReactFlow with ReactFlowProvider */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
        style={{ height: 'calc(100% - 80px)' }} // Adjust height to fit header
      >
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </ReactFlowProvider>
  </div>
);
};

export default ModelViz;
