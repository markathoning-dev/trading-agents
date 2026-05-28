export interface CardStats {
  risk_tolerance: number;
  volatility_preference: number;
  drawdown_penalty: number;
  trade_frequency: number;
}

export interface StrategyCard {
  id: string;
  name: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  mana_cost: number;
  description: string;
  flavor_text: string;
  stats: CardStats;
  reward_type: string | null;
  nodes: string[];
  prompt_modifier: string;
}

export interface Deck {
  id: string;
  name: string;
  card_ids: string[];
  mana_budget: number;
  total_mana?: number;
  is_valid?: boolean;
  errors?: string[];
  cards?: StrategyCard[];
  created_at?: string;
}

export const RARITY_COLORS: Record<string, string> = {
  common: '#9ca3af',
  rare: '#3b82f6',
  epic: '#a855f7',
  legendary: '#eab308',
};

export const RARITY_LABELS: Record<string, string> = {
  common: 'Common',
  rare: 'Rare',
  epic: 'Epic',
  legendary: 'Legendary',
};
