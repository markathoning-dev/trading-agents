import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { BacktestList } from './pages/BacktestList'
import { BacktestDetailPage } from './pages/BacktestDetail'
import { NewBacktest } from './pages/NewBacktest'
import { ModelCompare } from './pages/ModelCompare'
import { PinnTrain } from './pages/PinnTrain'
import { PinnGenerate } from './pages/PinnGenerate'
import { CardCollection } from './pages/CardCollection'
import { DeckBuilder } from './pages/DeckBuilder'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/app" element={<Dashboard />} />
          <Route path="/app/backtests" element={<BacktestList />} />
          <Route path="/app/backtests/new" element={<NewBacktest />} />
          <Route path="/app/backtests/:runId" element={<BacktestDetailPage />} />
          <Route path="/app/models/compare" element={<ModelCompare />} />
          <Route path="/app/pinn/train" element={<PinnTrain />} />
          <Route path="/app/pinn/generate" element={<PinnGenerate />} />
          <Route path="/app/cards" element={<CardCollection />} />
          <Route path="/app/decks" element={<DeckBuilder />} />
          <Route path="*" element={<Navigate to="/app" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}