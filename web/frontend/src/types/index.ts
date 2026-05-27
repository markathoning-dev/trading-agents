export interface BacktestResult {
  final_portfolio_value: number
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  cumulative_reward: number
  num_steps: number
  final_cash: number
  final_shares: number
}

export interface BacktestRun {
  id: number
  model_name: string
  data_source: string
  config: Record<string, unknown>
  status: string
  created_at: string | null
  result: BacktestResult | null
}

export interface BacktestStep {
  step: number
  price: number
  cash: number
  shares: number
  action: string
  portfolio_value: number
  reward: number
}

export interface BacktestDetail extends BacktestRun {
  steps: BacktestStep[]
}

export interface ModelCompareRow {
  model_name: string
  avg_return: number
  avg_sharpe: number
  avg_drawdown: number
  count: number
}

export interface PinnModel {
  id: number
  name: string
  pde_type: string
  architecture: Record<string, unknown>
  status: string
  created_at: string | null
}

export interface FormField {
  name: string
  type: string
  default: string | number
  options?: string[]
}

export interface PinnTrainForm {
  fields: FormField[]
}

export interface PinnGenerateForm {
  models: { id: number; name: string; pde_type: string }[]
  fields: FormField[]
}