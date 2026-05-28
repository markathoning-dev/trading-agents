import { useEffect, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { get } from '../api/client'
import type { BacktestDetail } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { StatCard } from '../components/StatCard'
import { TerminalTable } from '../components/TerminalTable'
import { Btn } from '../components/Btn'
import { createChart, ColorType, LineSeries } from 'lightweight-charts'

const CHART_COLORS = {
  background: '#0a0a0a',
  text: '#666666',
  grid: '#1a1a1a',
  green: '#00ff41',
  red: '#ff3333',
  blue: '#00d4ff',
}

export function BacktestDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [data, setData] = useState<BacktestDetail | null>(null)
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (runId) get<BacktestDetail>(`/backtests/${runId}`).then(setData).catch(console.error)
  }, [runId])

  useEffect(() => {
    if (!data?.steps?.length || !chartRef.current) return

    const chartEl = chartRef.current
    chartEl.innerHTML = ''

    const chart = createChart(chartEl, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_COLORS.background },
        textColor: CHART_COLORS.text,
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      width: chartEl.clientWidth,
      height: 280,
      crosshair: { mode: 0 },
      timeScale: { borderColor: CHART_COLORS.grid },
      rightPriceScale: { borderColor: CHART_COLORS.grid },
    })

    const portfolioData = data.steps.map((s, i) => ({
      time: i as unknown as import('lightweight-charts').Time,
      value: s.portfolio_value,
    }))

    const lineSeries = chart.addSeries(LineSeries, {
      color: CHART_COLORS.green,
      lineWidth: 2,
    })
    lineSeries.setData(portfolioData)

    chart.timeScale().fitContent()

    const handleResize = () => {
      if (chartEl) chart.applyOptions({ width: chartEl.clientWidth })
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  if (!data) {
    return (
      <div className="py-4">
        <p className="text-text-muted animate-flicker">Loading backtest data...</p>
      </div>
    )
  }

  return (
    <div className="py-4">
      <div className="flex items-center gap-4 mb-6 flex-wrap">
        <Link to="/app/backtests" className="text-text-muted text-[13px] hover:text-terminal-green">
          &larr; Back
        </Link>
        <h1 className="text-sm text-terminal-green tracking-wider">
          BACKTEST #{data.id}
        </h1>
        <span className="text-text-muted text-[13px]">{data.model_name}</span>
        <StatusBadge status={data.status} />
      </div>

      {data.result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              label="Return"
              value={`${(data.result.total_return * 100).toFixed(2)}%`}
              change={{ value: data.result.total_return * 100 }}
            />
            <StatCard label="Sharpe" value={data.result.sharpe_ratio.toFixed(2)} />
            <StatCard
              label="Max Drawdown"
              value={`${(data.result.max_drawdown * 100).toFixed(2)}%`}
              change={{ value: -(data.result.max_drawdown * 100) }}
            />
            <StatCard label="Steps" value={data.result.num_steps} />
          </div>

          <div className="mb-8">
            <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-3">
              Portfolio Value
            </div>
            <div
              ref={chartRef}
              className="bg-screen-bg border border-screen-border rounded overflow-hidden"
            />
          </div>

          <div className="mb-4 flex gap-3">
            <Link to="/app/backtests/new">
              <Btn variant="ghost">Re-run</Btn>
            </Link>
            <Link to="/app/models/compare">
              <Btn variant="ghost">Compare Models</Btn>
            </Link>
          </div>

          <div className="mb-3">
            <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-3">
              Trade Log
            </div>
            <TerminalTable
              columns={[
                { key: 'step', label: 'Step' },
                { key: 'price', label: 'Price', render: (s) => `$${s.price.toFixed(2)}` },
                {
                  key: 'action',
                  label: 'Action',
                  render: (s) => (
                    <span
                      className={
                        s.action === 'BUY'
                          ? 'text-terminal-cyan'
                          : s.action === 'SELL'
                            ? 'text-terminal-red'
                            : 'text-text-muted'
                      }
                    >
                      {s.action}
                    </span>
                  ),
                },
                {
                  key: 'portfolio_value',
                  label: 'Portfolio',
                  render: (s) => `$${s.portfolio_value.toFixed(2)}`,
                },
                {
                  key: 'reward',
                  label: 'Reward',
                  render: (s) => (
                    <span className={s.reward >= 0 ? 'text-terminal-green' : 'text-terminal-red'}>
                      {s.reward.toFixed(4)}
                    </span>
                  ),
                },
              ]}
              rows={data.steps}
            />
          </div>
        </>
      )}

      {!data.result && (
        <p className="text-text-muted text-[13px]">Status: {data.status}</p>
      )}
    </div>
  )
}