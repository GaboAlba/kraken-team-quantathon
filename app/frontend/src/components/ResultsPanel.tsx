import {
  Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import type { MethodResult, QaoaResult, RunRecord, RunResults } from '../types'

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
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="energy" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Legend />
          <ReferenceLine x={optimum.toFixed(2)} stroke="#d62728" strokeDasharray="4 4" />
          <Bar dataKey="classical" name={title} fill="#8c564b" opacity={0.7} />
          <Bar dataKey="qaoa" name="QAOA shots" fill="#1f77b4" opacity={0.7} />
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
  found: boolean
}) {
  return (
    <div className="card">
      <strong>{name}</strong>
      <div>best E = {best.toFixed(4)}</div>
      <div>gap {gap.toFixed(2)}% {found ? '🏆' : ''}</div>
      <div className="detail">{extra}</div>
    </div>
  )
}

export default function ResultsPanel({ results, run }: {
  results: RunResults
  run: RunRecord
}) {
  const m = results.methods
  const opt = results.optimum
  if (!opt) return null
  return (
    <section>
      <h2>Results</h2>
      <div className="cards">
        {m.brute_force && (
          <Card name="Brute force" best={m.brute_force.best_energy}
            gap={m.brute_force.gap_pct} found={m.brute_force.found_optimum}
            extra={`${m.brute_force.time_ms.toFixed(0)} ms`} />
        )}
        {m.greedy && (
          <Card name="Greedy" best={m.greedy.best_energy} gap={m.greedy.gap_pct}
            found={m.greedy.found_optimum}
            extra={`${m.greedy.energies.length} restarts`} />
        )}
        {m.goemans_williamson && (
          <Card name="Goemans-Williamson" best={m.goemans_williamson.best_energy}
            gap={m.goemans_williamson.gap_pct}
            found={m.goemans_williamson.found_optimum} extra="SDP + rounding" />
        )}
        {m.qaoa && (
          <Card name="QAOA (Helios)" best={m.qaoa.best_energy} gap={m.qaoa.gap_pct}
            found={m.qaoa.found_optimum}
            extra={`P(opt) ${(m.qaoa.p_optimal * 100).toFixed(1)}% · ${m.qaoa.shots} shots`} />
        )}
      </div>
      {m.brute_force && (
        <MethodChart title="Brute force (landscape)" method={m.brute_force}
          qaoa={m.qaoa} optimum={opt.energy} />
      )}
      {m.greedy && (
        <MethodChart title="Greedy restarts" method={m.greedy} qaoa={m.qaoa}
          optimum={opt.energy} />
      )}
      {m.goemans_williamson && (
        <MethodChart title="GW roundings" method={m.goemans_williamson}
          qaoa={m.qaoa} optimum={opt.energy} />
      )}
      <h3>Log</h3>
      <pre className="log">{run.log.join('\n')}</pre>
    </section>
  )
}
