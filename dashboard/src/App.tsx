import { Route, Routes } from 'react-router-dom';
import { OverviewPage } from './pages/OverviewPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { DriftPage } from './pages/DriftPage';
import { PredictionsPage } from './pages/PredictionsPage';
import { PipelinePage } from './pages/PipelinePage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<OverviewPage />} />
      <Route path="/experiments" element={<ExperimentsPage />} />
      <Route path="/drift" element={<DriftPage />} />
      <Route path="/predictions" element={<PredictionsPage />} />
      <Route path="/pipeline" element={<PipelinePage />} />
    </Routes>
  );
}
