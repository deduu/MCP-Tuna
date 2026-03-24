import { cn } from '@/lib/utils'

interface LossChartProps {
  data: Array<{ step: number; loss?: number; evalLoss?: number; learning_rate?: number }>
  className?: string
}

export function LossChart({ data, className }: LossChartProps) {
  const chartData = data.filter(
    (point) =>
      Number.isFinite(point.step) &&
      (Number.isFinite(point.loss) || Number.isFinite(point.evalLoss)),
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

  const width = 1000
  const height = 160
  const padLeft = 56
  const padRight = 18
  const padTop = 14
  const padBottom = 28

  const chartW = width - padLeft - padRight
  const chartH = height - padTop - padBottom

  const steps = chartData.map((d) => d.step)
  const losses = chartData.flatMap((d) => [d.loss, d.evalLoss]).filter(Number.isFinite) as number[]
  const minStep = Math.min(...steps)
  const maxStep = Math.max(...steps)
  const minLoss = Math.min(...losses)
  const maxLoss = Math.max(...losses)
  const uniqueStepCount = new Set(steps).size
  const useIndexForX = uniqueStepCount < 2
  const xRange = Math.max(chartData.length - 1, 1)

  const lossRange = maxLoss - minLoss
  const lossPad = lossRange > 0 ? lossRange * 0.18 : Math.max(maxLoss * 0.35, 0.1)
  const yMin = Math.max(0, minLoss - lossPad)
  const yMax = Math.max(maxLoss + lossPad, yMin + 0.25)
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

  const chartPoints = chartData.map((d, index) => ({
    x: toX(d.step, index),
    trainY: d.loss != null ? toY(d.loss) : undefined,
    evalY: d.evalLoss != null ? toY(d.evalLoss) : undefined,
  }))
  const trainStrokeColor = 'var(--color-ns-finetune)'
  const evalStrokeColor = 'var(--color-warning, #f59e0b)'

  function buildSmoothPath(coords: Array<{ x: number; y: number }>) {
    if (coords.length === 0) return ''
    if (coords.length === 1) return `M ${coords[0].x} ${coords[0].y}`
    if (coords.length === 2) return `M ${coords[0].x} ${coords[0].y} L ${coords[1].x} ${coords[1].y}`

    let path = `M ${coords[0].x} ${coords[0].y}`
    for (let i = 0; i < coords.length - 1; i += 1) {
      const current = coords[i]
      const next = coords[i + 1]
      const midX = (current.x + next.x) / 2
      const midY = (current.y + next.y) / 2
      path += ` Q ${current.x} ${current.y} ${midX} ${midY}`
      if (i === coords.length - 2) {
        path += ` T ${next.x} ${next.y}`
      }
    }
    return path
  }

  const trainPoints = chartPoints.flatMap((point, index) =>
    point.trainY == null ? [] : [{ x: point.x, y: point.trainY, key: `${chartData[index].step}-${index}` }],
  )
  const evalPoints = chartPoints.flatMap((point, index) =>
    point.evalY == null ? [] : [{ x: point.x, y: point.evalY, key: `${chartData[index].step}-${index}` }],
  )
  const trainPath = buildSmoothPath(trainPoints)
  const evalPath = buildSmoothPath(evalPoints)
  const hasEvalSeries = evalPoints.length > 0

  const gridLines = 4
  const gridYValues = Array.from({ length: gridLines }, (_, i) => {
    const frac = i / (gridLines - 1)
    return yMin + frac * yRange
  })

  return (
    <div className={cn('w-full', className)}>
      {hasEvalSeries && (
        <div className="mb-2 flex items-center gap-4 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: trainStrokeColor }}
            />
            Train loss
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: evalStrokeColor }}
            />
            Validation loss
          </span>
        </div>
      )}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height: 160 }}
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

        {/* Loss path */}
        {trainPoints.length > 1 && (
          <path
            fill="none"
            d={trainPath}
            strokeWidth={2.25}
            strokeLinejoin="round"
            strokeLinecap="round"
            style={{ stroke: trainStrokeColor }}
          />
        )}

        {evalPoints.length > 1 && (
          <path
            fill="none"
            d={evalPath}
            strokeWidth={2.25}
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray="5 4"
            style={{ stroke: evalStrokeColor }}
          />
        )}

        {trainPoints.map((point) => (
          <circle
            key={`train-${point.key}`}
            cx={point.x}
            cy={point.y}
            r={trainPoints.length === 1 ? 4 : 2.5}
            style={{ fill: trainStrokeColor }}
          />
        ))}

        {evalPoints.map((point) => (
          <circle
            key={`eval-${point.key}`}
            cx={point.x}
            cy={point.y}
            r={evalPoints.length === 1 ? 4 : 2.5}
            style={{ fill: evalStrokeColor }}
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
