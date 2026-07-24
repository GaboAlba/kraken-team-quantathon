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

export interface SubgridInfo {
  valid: boolean
  reason: string | null
  nodes: string[]
  edges: GridEdge[]
  adjacent: string[]
}

export type StageState = 'pending' | 'running' | 'done' | 'error'

export interface Stage {
  name: string
  state: StageState
  detail: string
}

export interface MethodResult {
  best_energy: number
  gap_pct: number
  time_ms: number
  found_optimum: boolean
  energies: number[]
}

export interface QaoaResult {
  best_energy: number
  gap_pct: number
  mean_energy: number
  found_optimum: boolean
  p_optimal: number
  first_optimal_shot: number | null
  shots: number
  energies: number[]
  gamma: number
  beta: number
  job_id: string
}

export interface RunResults {
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
}
