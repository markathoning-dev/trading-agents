import React from 'react';
import { StrategyCard as StrategyCardType, RARITY_COLORS } from '../types/cards';

interface StrategyCardProps {
  card: StrategyCardType;
  onClick?: (card: StrategyCardType) => void;
  selected?: boolean;
  compact?: boolean;
}

const StatBar: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
    <span style={{ fontSize: '11px', color: '#9ca3af', width: '100px' }}>{label}</span>
    <div style={{ flex: 1, height: '8px', background: '#1f2937', borderRadius: '4px', overflow: 'hidden' }}>
      <div
        style={{
          width: `${value * 100}%`,
          height: '100%',
          background: `linear-gradient(90deg, #3b82f6, #8b5cf6)`,
          borderRadius: '4px',
        }}
      />
    </div>
    <span style={{ fontSize: '11px', color: '#9ca3af', width: '30px', textAlign: 'right' }}>
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
      style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        border: `2px solid ${selected ? '#3b82f6' : rarityColor}`,
        borderRadius: '12px',
        padding: compact ? '12px' : '16px',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        boxShadow: selected
          ? '0 0 20px rgba(59, 130, 246, 0.3)'
          : card.rarity === 'legendary'
          ? '0 0 15px rgba(234, 179, 8, 0.2)'
          : 'none',
        minWidth: compact ? '180px' : '240px',
        maxWidth: compact ? '200px' : '280px',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0, fontSize: compact ? '14px' : '16px', color: '#f3f4f6' }}>
          {card.name}
        </h3>
        <div
          style={{
            background: rarityColor,
            color: card.rarity === 'legendary' ? '#000' : '#fff',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: 'bold',
            textTransform: 'uppercase',
          }}
        >
          {card.rarity}
        </div>
      </div>

      {/* Mana Cost */}
      <div style={{ display: 'flex', gap: '4px', marginBottom: '8px' }}>
        {Array.from({ length: card.mana_cost }).map((_, i) => (
          <span key={i} style={{ color: '#60a5fa', fontSize: '16px' }}>
            ⚡
          </span>
        ))}
        {Array.from({ length: Math.max(0, 5 - card.mana_cost) }).map((_, i) => (
          <span key={i} style={{ color: '#374151', fontSize: '16px' }}>
            ⚡
          </span>
        ))}
      </div>

      {/* Description */}
      {!compact && (
        <p style={{ fontSize: '12px', color: '#d1d5db', margin: '0 0 8px 0', lineHeight: '1.4' }}>
          {card.description}
        </p>
      )}

      {/* Flavor Text */}
      {!compact && (
        <p style={{ fontSize: '11px', color: '#6b7280', margin: '0 0 12px 0', fontStyle: 'italic' }}>
          "{card.flavor_text}"
        </p>
      )}

      {/* Stats */}
      {!compact && (
        <div style={{ marginBottom: '12px' }}>
          <StatBar label="Risk Tolerance" value={card.stats.risk_tolerance} />
          <StatBar label="Volatility Pref" value={card.stats.volatility_preference} />
          <StatBar label="Drawdown Penalty" value={card.stats.drawdown_penalty} />
          <StatBar label="Trade Frequency" value={card.stats.trade_frequency} />
        </div>
      )}

      {/* Reward Type */}
      {card.reward_type && (
        <div
          style={{
            background: '#1f2937',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#9ca3af',
            marginBottom: '8px',
          }}
        >
          Reward: <span style={{ color: '#60a5fa' }}>{card.reward_type}</span>
        </div>
      )}

      {/* Nodes */}
      {card.nodes.length > 0 && !compact && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {card.nodes.map((node) => (
            <span
              key={node}
              style={{
                background: '#374151',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '10px',
                color: '#9ca3af',
              }}
            >
              {node}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
