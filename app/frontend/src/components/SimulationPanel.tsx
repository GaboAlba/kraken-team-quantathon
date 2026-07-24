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
  onStop: () => void
}

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m ${String(Math.floor(s % 60)).padStart(2, '0')}s`
}

function marker(state: string): string {
  if (state === 'done') return '✓'
  if (state === 'running') return '◌'
  if (state === 'error') return '✗'
  if (state === 'skipped') return '–'
  if (state === 'cancelled') return '■'
  return '·'
}

export default function SimulationPanel({ disabled, busy, run, error, onRun, onStop }: Props) {
  return (
    <section>
      <h2 className="section-title">
        Simulation
        {run && (
          <span className="count">
            {run.progress_pct}% · {fmtTime(run.elapsed_s)}
          </span>
        )}
        {run && <span className={`status-chip ${run.status}`}>{run.status}</span>}
      </h2>
      <div className="run-row">
        <button className="run-btn" disabled={disabled || busy} onClick={onRun}>
          {busy ? 'Running…' : 'Run simulation'}
        </button>
        {busy && (
          <button className="stop-btn" onClick={onStop} title="cancel the run">
            ■ Stop
          </button>
        )}
      </div>
      {error && <p className="invalid">{error}</p>}
      {run && (
        <>
          <div className="progress">
            <div className="progress-fill" style={{ width: `${run.progress_pct}%` }} />
          </div>
          <ul className="stages">
            {run.stages.map((s) => (
              <li key={s.name} className={`stage ${s.state}`}>
                <span className="marker">{marker(s.state)}</span>
                {STAGE_LABELS[s.name] ?? s.name}
                {s.detail && <span className="detail">{s.detail}</span>}
                {s.elapsed_s !== null && (
                  <span className="stage-time">{fmtTime(s.elapsed_s)}</span>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
