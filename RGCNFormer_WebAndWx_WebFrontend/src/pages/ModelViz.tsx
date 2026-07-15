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
