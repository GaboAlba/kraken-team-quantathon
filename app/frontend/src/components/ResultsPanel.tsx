import {
  Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import type { MethodResult, QaoaResult, RunRecord, RunResults } from '../types'

const CHART = {
  grid: '#223140',
  tick: '#7e909c',
  classical: '#c9a26d',
  qaoa: '#4aa3ff',
  optimum: '#3fd77f',
  tooltipBg: '#16212a',
}

interface Bin {
  energy: string
  classical: number
  qaoa: number
}

function bins(classical: number[], qaoa: number[], nBins = 24): Bin[] {
  let lo = Infinity
  let hi = -Infinity
  for (const e of classical) {
    if (e < lo) lo = e
    if (e > hi) hi = e
  }
  for (const e of qaoa) {
    if (e < lo) lo = e
    if (e > hi) hi = e
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0
    hi = 0
  }
  const width = (hi - lo) / nBins || 1
  const rows: Bin[] = []
  for (let i = 0; i < nBins; i++) {
    const a = lo + i * width
    const inBin = (e: number) => e >= a && (i === nBins - 1 ? e <= a + width : e < a + width)
    rows.push({
      energy: (a + width / 2).toFixed(2),
      classical: classical.length ? classical.filter(inBin).length / classical.length : 0,
      qaoa: qaoa.length ? qaoa.filter(inBin).length / qaoa.length : 0,
    })
  }
  return rows
}

function MethodChart({ title, method, qaoa, optimum }: {
  title: string
  method: MethodResult
  qaoa: QaoaResult | null
  optimum: number
}) {
  const data = bins(method.energies, qaoa?.energies ?? [])
  return (
    <div className="chart-block">
      <h4>{title} vs QAOA</h4>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data} barCategoryGap={0}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis dataKey="energy" tick={{ fontSize: 10, fill: CHART.tick }}
            stroke={CHART.grid} />
          <YAxis tick={{ fontSize: 10, fill: CHART.tick }} stroke={CHART.grid} />
          <Tooltip
            contentStyle={{
              background: CHART.tooltipBg,
              border: `1px solid ${CHART.grid}`,
              borderRadius: 5,
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: 11,
            }}
            labelStyle={{ color: CHART.tick }}
            itemStyle={{ color: '#d8e2e9' }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: CHART.tick }} />
          <ReferenceLine x={optimum.toFixed(2)} stroke={CHART.optimum}
            strokeDasharray="4 4" />
          <Bar dataKey="classical" name={title} fill={CHART.classical} opacity={0.8} />
          <Bar dataKey="qaoa" name="QAOA shots" fill={CHART.qaoa} opacity={0.75} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function Card({ name, best, gap, extra, found }: {
  name: string
  best: number
  gap: number
  extra: string
  found: boolean | null
}) {
  return (
    <div className="card">
      {found === true && <span className="badge">OPTIMAL</span>}
      <div className="method">{name}</div>
      <div className="value">
        {best.toFixed(4)}
        <span className="unit">E</span>
      </div>
      <div className={`gap ${gap <= 1e-9 ? 'zero' : ''}`}>
        gap {gap.toFixed(2)}%
      </div>
      <div className="extra">{extra}</div>
    </div>
  )
}

export default function ResultsPanel({ results, run }: {
  results: RunResults
  run: RunRecord
}) {
  const m = results.methods
  const ref = results.reference
  if (!ref) return null
  const refLabel = ref.type === 'exact' ? 'exact optimum' : 'SDP certified lower bound'
  return (
    <section>
      <h2 className="section-title">
        Results
        {results.tier && <span className={`tier tier-${results.tier}`}>{results.tier}</span>}
      </h2>
      <p className="ref-note">
        reference: {refLabel} = <b>{ref.energy.toFixed(4)}</b>
      </p>
      {results.best_partition && (
        <p className="ref-note">
          map: <span className="dot zone-a" /> zone A ({results.best_partition.A.length}) ·{' '}
          <span className="dot zone-b" /> zone B ({results.best_partition.B.length}) ·{' '}
          <span className="dot cut" /> cut lines — best found by{' '}
          <b>{results.best_partition.method}</b>
        </p>
      )}
      <div className="cards">
        {m.brute_force && (
          <Card name="Brute force" best={m.brute_force.best_energy}
            gap={m.brute_force.gap_pct} found={m.brute_force.found_optimum}
            extra={`${(m.brute_force.time_ms / 1000).toFixed(2)} s${m.brute_force.n_states ? ` · ${m.brute_force.n_states.toLocaleString()} states` : ''}`} />
        )}
        {m.greedy && (
          <Card name="Greedy" best={m.greedy.best_energy} gap={m.greedy.gap_pct}
            found={m.greedy.found_optimum}
            extra={`${(m.greedy.time_ms / 1000).toFixed(2)} s · ${m.greedy.energies.length} restarts`} />
        )}
        {m.goemans_williamson && (
          <Card name="Goemans-Williamson" best={m.goemans_williamson.best_energy}
            gap={m.goemans_williamson.gap_pct}
            found={m.goemans_williamson.found_optimum}
            extra={`${(m.goemans_williamson.time_ms / 1000).toFixed(2)} s · SDP + rounding`} />
        )}
        {m.qaoa && (
          <Card name="QAOA · Helios" best={m.qaoa.best_energy} gap={m.qaoa.gap_pct}
            found={m.qaoa.found_optimum}
            extra={`queued ${m.qaoa.queued_s.toFixed(1)} s · run ${m.qaoa.running_s.toFixed(1)} s
${m.qaoa.p_optimal !== null ? `P(opt) ${(m.qaoa.p_optimal * 100).toFixed(1)}% · ` : ''}${m.qaoa.shots} shots`} />
        )}
      </div>
      {m.brute_force && (
        <MethodChart title="Brute force (landscape)" method={m.brute_force}
          qaoa={m.qaoa} optimum={ref.energy} />
      )}
      {m.greedy && (
        <MethodChart title="Greedy restarts" method={m.greedy} qaoa={m.qaoa}
          optimum={ref.energy} />
      )}
      {m.goemans_williamson && (
        <MethodChart title="GW roundings" method={m.goemans_williamson}
          qaoa={m.qaoa} optimum={ref.energy} />
      )}
      <h2 className="section-title">Log</h2>
      <pre className="log">{run.log.join('\n')}</pre>
    </section>
  )
}
