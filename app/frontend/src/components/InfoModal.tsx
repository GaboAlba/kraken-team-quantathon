import { useEffect, useState } from 'react'
import { fetchConfig } from '../api'
import type { AppConfig } from '../types'

interface Props {
  onClose: () => void
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="info-row">
      <span className="info-key">{k}</span>
      <span className="info-val">{v}</span>
    </div>
  )
}

export default function InfoModal({ onClose }: Props) {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchConfig().then(setConfig).catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Algorithm configuration</h3>
          <button className="icon-btn" onClick={onClose} aria-label="close">×</button>
        </div>
        {error && <p className="invalid">{error}</p>}
        {config && (
          <div className="modal-body">
            <h4>QAOA (quantum)</h4>
            <Row k="layers (p)" v={String(config.qaoa.layers)} />
            <Row k="shots" v={String(config.qaoa.default_shots)} />
            <Row k="qubits" v="1 per selected substation" />
            <Row k="angle search" v={config.qaoa.angle_search} />
            <Row k="circuit" v={config.qaoa.circuit} />

            <h4>QUBO</h4>
            <Row k="weight scheme" v={config.qubo.weight_scheme} />
            <Row k="objective" v={config.qubo.objective} />
            <Row k="reference voltage" v={`${config.qubo.reference_voltage_kv} kV`} />
            {config.qubo.generator_spread_factor !== null && (
              <Row k="generator spread penalty" v={`${config.qubo.generator_spread_factor} × w_max per pair`} />
            )}
            {config.qubo.balance_factor !== null && (
              <Row k="balance penalty" v={`${config.qubo.balance_factor} × w_max`} />
            )}

            <h4>Classical baselines</h4>
            <Row k="greedy" v={`${config.classical.greedy_restarts} restarts (seeds 0–${config.classical.greedy_restarts - 1})`} />
            <Row k="Goemans-Williamson" v={`${config.classical.gw_rounding_trials} hyperplane roundings, seed ${config.classical.gw_seed}`} />
            <Row k="SDP solver" v={config.classical.sdp_solver} />
            <Row k="brute force" v={`vectorized enumeration up to n=${config.scaling.brute_force_max_n}`} />

            <h4>Nexus (quantum backend)</h4>
            <Row k="device" v={config.nexus.device} />
            <Row k="project" v={config.nexus.project} />
            <Row k="emulation" v={config.nexus.emulator} />
            <Row k="poll timeout" v={`${config.nexus.poll_timeout_s / 60} min`} />

            <h4>Scaling</h4>
            <Row k="exact angles" v={`up to ${config.scaling.exact_angles_max_n} qubits (statevector)`} />
            <Row k="exact optimum" v={`up to ${config.scaling.brute_force_max_n} nodes (2^n enumeration)`} />
            <Row k="initial subgrid" v={`${config.scaling.initial_nodes.length} locked nodes (Guanacaste North)`} />
          </div>
        )}
      </div>
    </div>
  )
}
