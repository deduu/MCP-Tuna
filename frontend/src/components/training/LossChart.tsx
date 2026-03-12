import { cn } from '@/lib/utils'

interface LossChartProps {
  data: Array<{ step: number; loss: number; learning_rate?: number }>
  className?: string
}

export function LossChart({ data, className }: LossChartProps) {
  const chartData = data.filter(
    (point) => Number.isFinite(point.step) && Number.isFinite(point.loss),
  )

  if (chartData.length === 0) {
    return (
      <div
        className={cn(
          'flex h-[120px] w-full items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground',
          className,
        )}
      >
        No data
      </div>
    )
  }

  const width = 400
  const height = 120
  const padLeft = 45
  const padRight = 10
  const padTop = 8
  const padBottom = 20

  const chartW = width - padLeft - padRight
  const chartH = height - padTop - padBottom

  const steps = chartData.map((d) => d.step)
  const losses = chartData.map((d) => d.loss)
  const minStep = Math.min(...steps)
  const maxStep = Math.max(...steps)
  const minLoss = Math.min(...losses)
  const maxLoss = Math.max(...losses)
  const uniqueStepCount = new Set(steps).size
  const useIndexForX = uniqueStepCount < 2
  const xRange = Math.max(chartData.length - 1, 1)

  const lossRange = maxLoss - minLoss || 1
  const lossPad = lossRange * 0.1
  const yMin = minLoss - lossPad
  const yMax = maxLoss + lossPad
  const yRange = yMax - yMin

  const stepRange = maxStep - minStep || 1

  function toX(step: number, index: number) {
    if (useIndexForX) {
      return padLeft + (index / xRange) * chartW
    }
    return padLeft + ((step - minStep) / stepRange) * chartW
  }

  function toY(loss: number) {
    return padTop + chartH - ((loss - yMin) / yRange) * chartH
  }

  const points = chartData.map((d, index) => `${toX(d.step, index)},${toY(d.loss)}`).join(' ')
  const strokeColor = 'var(--color-ns-finetune)'

  const gridLines = 3
  const gridYValues = Array.from({ length: gridLines }, (_, i) => {
    const frac = i / (gridLines - 1)
    return yMin + frac * yRange
  })

  return (
    <div className={cn('w-full', className)}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height: 120 }}
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        {gridYValues.map((val, i) => {
          const y = toY(val)
          return (
            <g key={i}>
              <line
                x1={padLeft}
                y1={y}
                x2={width - padRight}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.15}
                strokeDasharray="4 3"
              />
              <text
                x={padLeft - 4}
                y={y + 3}
                textAnchor="end"
                fontSize={8}
                fill="currentColor"
                fillOpacity={0.5}
              >
                {val.toFixed(3)}
              </text>
            </g>
          )
        })}

        {/* Loss polyline */}
        {chartData.length > 1 && (
          <polyline
            fill="none"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            points={points}
            style={{ stroke: strokeColor }}
          />
        )}

        {chartData.map((point, index) => (
          <circle
            key={`${point.step}-${index}`}
            cx={toX(point.step, index)}
            cy={toY(point.loss)}
            r={chartData.length === 1 ? 3.5 : 2}
            style={{ fill: strokeColor }}
          />
        ))}

        {/* X axis labels */}
        <text
          x={padLeft}
          y={height - 4}
          textAnchor="start"
          fontSize={8}
          fill="currentColor"
          fillOpacity={0.5}
        >
          {minStep}
        </text>
        <text
          x={width - padRight}
          y={height - 4}
          textAnchor="end"
          fontSize={8}
          fill="currentColor"
          fillOpacity={0.5}
        >
          {maxStep}
        </text>
      </svg>
    </div>
  )
}
