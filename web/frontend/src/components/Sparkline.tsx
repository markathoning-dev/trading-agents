export function Sparkline({
  data,
  width = 60,
  height = 16,
  className = '',
}: {
  data: number[]
  width?: number
  height?: number
  className?: string
}) {
  if (data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const padding = 1
  const chartW = width - padding * 2
  const chartH = height - padding * 2

  const isPositive = data[data.length - 1] >= data[0]
  const strokeColor = isPositive ? '#00ff41' : '#ff3333'

  const points = data
    .map((v, i) => {
      const x = padding + (i / (data.length - 1)) * chartW
      const y = padding + chartH - ((v - min) / range) * chartH
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg width={width} height={height} className={className} viewBox={`0 0 ${width} ${height}`}>
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke="#1a1a1a"
        strokeWidth={0.5}
      />
      <polyline points={points} fill="none" stroke={strokeColor} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}