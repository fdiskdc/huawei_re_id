import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { LanguageProvider } from './lib/i18n/LanguageContext';
import MainPage from './pages/MainPage';
import ClassificationViz from './pages/ClassificationViz';
import AttentionViz from './pages/AttentionViz';
import GcnViz from './pages/GcnViz';
import TargetGcnViz from './pages/TargetGcnViz';
import IntegratedGradientsViz from './pages/IntegratedGradientsViz';
import ModelViz from './pages/ModelViz';
import ResultsPage from './pages/ResultsPage';
import ReidViz from './pages/ReidViz';
import VizLayout from './components/VizLayout';
import './App.css';

// Create a client for TanStack Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 1000 * 60 * 5, // 5 minutes
    },
  },
});

const App: React.FC = () => {
  return (
    <LanguageProvider>
      <QueryClientProvider client={queryClient}>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route element={<VizLayout />}>
            <Route path="/classification" element={<ClassificationViz />} />
            <Route path="/attention" element={<AttentionViz />} />
            <Route path="/gcn" element={<GcnViz />} />
            <Route path="/target-gcn" element={<TargetGcnViz />} />
            <Route path="/integrated-gradients" element={<IntegratedGradientsViz />} />
            <Route path="/model-viz" element={<ModelViz />} />
            <Route path="/reid" element={<ReidViz />} />
          </Route>
          <Route path="/results/:jobId" element={<ResultsPage />} />
        </Routes>
      </QueryClientProvider>
    </LanguageProvider>
  );
};

export default App;
