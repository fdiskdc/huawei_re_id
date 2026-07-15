import { Button, Card, Space, Typography } from 'antd';
import { useNavigate } from 'react-router-dom';

const { Title, Paragraph } = Typography;

export default function LandingPage() {
  const navigate = useNavigate();
  return <main className="mode-landing">
    <Title>mRModN</Title>
    <Paragraph>Choose the experience that fits your workflow.</Paragraph>
    <Space size="large" wrap>
      <Card title="Classic" className="mode-card"><Paragraph>The original focused prediction and explainability workflow.</Paragraph><Button type="primary" onClick={() => navigate('/classic')}>Open Classic</Button></Card>
      <Card title="NextGen" className="mode-card"><Paragraph>Workspace, comparison, localization, UMAP and attention tools.</Paragraph><Button type="primary" onClick={() => navigate('/nextgen')}>Open NextGen</Button></Card>
    </Space>
  </main>;
}
