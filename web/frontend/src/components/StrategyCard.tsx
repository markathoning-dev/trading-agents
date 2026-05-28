import React from 'react';
import { StrategyCard as StrategyCardType, RARITY_COLORS } from '../types/cards';

interface StrategyCardProps {
  card: StrategyCardType;
  onClick?: (card: StrategyCardType) => void;
  selected?: boolean;
  compact?: boolean;
}

const StatBar: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div className="flex items-center gap-2 mb-1">
    <span className="text-[11px] text-text-muted w-[100px]">{label}</span>
    <div className="flex-1 h-2 bg-screen-bg rounded overflow-hidden">
      <div
        className="h-full rounded"
        style={{
          width: `${value * 100}%`,
          background: 'linear-gradient(90deg, var(--color-terminal-cyan), var(--color-terminal-green))',
        }}
      />
    </div>
    <span className="text-[11px] text-text-dim w-[30px] text-right tabular-nums">
      {value.toFixed(1)}
    </span>
  </div>
);

export const StrategyCardComponent: React.FC<StrategyCardProps> = ({
  card,
  onClick,
  selected = false,
  compact = false,
}) => {
  const rarityColor = RARITY_COLORS[card.rarity] || '#9ca3af';

  return (
    <div
      onClick={() => onClick?.(card)}
      className="transition-all duration-200"
      style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        border: `2px solid ${selected ? '#00d4ff' : rarityColor}`,
        borderRadius: '12px',
        padding: compact ? '12px' : '16px',
        cursor: onClick ? 'pointer' : 'default',
        boxShadow: selected
          ? '0 0 20px rgba(0, 212, 255, 0.3)'
          : card.rarity === 'legendary'
            ? '0 0 15px rgba(255, 215, 0, 0.2)'
            : 'none',
        minWidth: compact ? '180px' : '240px',
        maxWidth: compact ? '200px' : '280px',
      }}
    >
      <div className="flex justify-between items-center mb-2">
        <h3 className={`${compact ? 'text-[14px]' : 'text-base'} text-text-primary m-0`}>
          {card.name}
        </h3>
        <span
          className="text-[10px] font-bold uppercase px-2 py-0.5 rounded"
          style={{
            backgroundColor: rarityColor,
            color: card.rarity === 'legendary' ? '#000' : '#fff',
          }}
        >
          {card.rarity}
        </span>
      </div>

      <div className="flex gap-1 mb-2">
        {Array.from({ length: card.mana_cost }).map((_, i) => (
          <span key={i} className="text-terminal-cyan text-base">{'\u26A1'}</span>
        ))}
        {Array.from({ length: Math.max(0, 5 - card.mana_cost) }).map((_, i) => (
          <span key={i} className="text-text-dim text-base">{'\u26A1'}</span>
        ))}
      </div>

      {!compact && (
        <p className="text-[12px] text-text-primary leading-relaxed mb-2 m-0">
          {card.description}
        </p>
      )}

      {!compact && (
        <p className="text-[11px] text-text-muted italic mb-3 m-0">
          "{card.flavor_text}"
        </p>
      )}

      {!compact && (
        <div className="mb-3">
          <StatBar label="Risk Tolerance" value={card.stats.risk_tolerance} />
          <StatBar label="Volatility Pref" value={card.stats.volatility_preference} />
          <StatBar label="Drawdown Penalty" value={card.stats.drawdown_penalty} />
          <StatBar label="Trade Frequency" value={card.stats.trade_frequency} />
        </div>
      )}

      {card.reward_type && (
        <div className="bg-screen-elevated px-2 py-1 rounded text-[11px] text-text-muted mb-2">
          Reward: <span className="text-terminal-cyan">{card.reward_type}</span>
        </div>
      )}

      {card.nodes.length > 0 && !compact && (
        <div className="flex flex-wrap gap-1">
          {card.nodes.map((node) => (
            <span
              key={node}
              className="bg-screen-elevated border border-screen-border px-1.5 py-0.5 rounded text-[10px] text-text-muted"
            >
              {node}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};