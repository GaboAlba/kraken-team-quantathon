import type { RunRecord } from '../types'

const STAGE_LABELS: Record<string, string> = {
  build_qubo: 'Build QUBO',
  brute_force: 'Brute force',
  greedy: 'Greedy',
  goemans_williamson: 'Goemans-Williamson',
  qaoa_angles: 'QAOA angles',
  nexus_job: 'Nexus job (Helios)',
  analysis: 'Analysis',
}

interface Props {
  disabled: boolean
  busy: boolean
  run: RunRecord | null
  error: string | null
  onRun: () => void
}

export default function SimulationPanel({ disabled, busy, run, error, onRun }: Props) {
  return (
    <section>
      <h2>Simulation</h2>
      <button className="run-btn" disabled={disabled || busy} onClick={onRun}>
        {busy ? 'Running…' : 'Run simulation'}
      </button>
      {error && <p className="invalid">{error}</p>}
      {run && (
        <>
          <div className="progress">
            <div className="progress-fill" style={{ width: `${run.progress_pct}%` }} />
          </div>
          <ul className="stages">
            {run.stages.map((s) => (
              <li key={s.name} className={`stage ${s.state}`}>
                {s.state === 'done' ? '✓' : s.state === 'running' ? '⟳' : s.state === 'error' ? '✗' : '·'}{' '}
                {STAGE_LABELS[s.name] ?? s.name}
                {s.detail && <span className="detail"> — {s.detail}</span>}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
