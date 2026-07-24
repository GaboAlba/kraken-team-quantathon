export interface GridNode {
  id: string
  name: string
  lat: number
  lon: number
  is_initial: boolean
}

export interface GridEdge {
  u: string
  v: string
  voltage: number | null
  weight: number
}

export interface Plant {
  name: string
  technology: string | null
  mw: number
  lat: number
  lon: number
  substation: string | null
}

export interface GridPayload {
  nodes: GridNode[]
  edges: GridEdge[]
  plants: Plant[]
}

export type Tier = 'exact' | 'heuristic' | 'classical'

export interface SubgridInfo {
  valid: boolean
  reason: string | null
  nodes: string[]
  edges: GridEdge[]
  adjacent: string[]
  tier: Tier
}

export type StageState = 'pending' | 'running' | 'done' | 'error' | 'skipped'

export interface Stage {
  name: string
  state: StageState
  detail: string
  elapsed_s: number | null
}

export interface MethodResult {
  best_energy: number
  gap_pct: number
  time_ms: number
  found_optimum: boolean | null
  energies: number[]
  n_states?: number
}

export interface QaoaResult {
  best_energy: number
  gap_pct: number
  mean_energy: number
  found_optimum: boolean | null
  p_optimal: number | null
  first_optimal_shot: number | null
  shots: number
  energies: number[]
  gamma: number
  beta: number
  job_id: string
  queued_s: number
  running_s: number
}

export interface RunResults {
  tier: Tier | null
  reference: { type: 'exact' | 'sdp_bound'; energy: number } | null
  sdp_bound_energy?: number
  optimum: { energy: number; partition: { A: string[]; B: string[] } } | null
  methods: {
    brute_force: MethodResult | null
    greedy: MethodResult | null
    goemans_williamson: MethodResult | null
    qaoa: QaoaResult | null
  }
}

export type RunStatus = 'running' | 'done' | 'error'

export interface RunRecord {
  id: string
  status: RunStatus
  stages: Stage[]
  log: string[]
  results: RunResults | null
  progress_pct: number
  elapsed_s: number
}
