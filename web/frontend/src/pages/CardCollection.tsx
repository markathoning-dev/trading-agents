import React, { useEffect, useState } from 'react';
import { listCards } from '../api/cards';
import { StrategyCard, RARITY_COLORS } from '../types/cards';
import { StrategyCardComponent } from '../components/StrategyCard';

export const CardCollection: React.FC = () => {
  const [cards, setCards] = useState<StrategyCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [selectedCard, setSelectedCard] = useState<StrategyCard | null>(null);

  useEffect(() => {
    listCards()
      .then(setCards)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const filteredCards = filter === 'all'
    ? cards
    : cards.filter((c) => c.rarity === filter);

  const rarityCounts = cards.reduce((acc, card) => {
    acc[card.rarity] = (acc[card.rarity] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  if (loading) {
    return <div style={{ color: '#9ca3af', padding: '20px' }}>Loading cards...</div>;
  }

  if (error) {
    return <div style={{ color: '#ef4444', padding: '20px' }}>Error: {error}</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1 style={{ color: '#f3f4f6', marginBottom: '20px' }}>Strategy Card Collection</h1>

      {/* Filters */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
        <button
          onClick={() => setFilter('all')}
          style={{
            background: filter === 'all' ? '#3b82f6' : '#1f2937',
            color: filter === 'all' ? '#fff' : '#9ca3af',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
        >
          All ({cards.length})
        </button>
        {['common', 'rare', 'epic', 'legendary'].map((rarity) => (
          <button
            key={rarity}
            onClick={() => setFilter(rarity)}
            style={{
              background: filter === rarity ? RARITY_COLORS[rarity] : '#1f2937',
              color: filter === rarity ? (rarity === 'legendary' ? '#000' : '#fff') : '#9ca3af',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              textTransform: 'capitalize',
            }}
          >
            {rarity} ({rarityCounts[rarity] || 0})
          </button>
        ))}
      </div>

      {/* Card Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
        {filteredCards.map((card) => (
          <StrategyCardComponent
            key={card.id}
            card={card}
            onClick={setSelectedCard}
            selected={selectedCard?.id === card.id}
          />
        ))}
      </div>

      {/* Card Detail Modal */}
      {selectedCard && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
          onClick={() => setSelectedCard(null)}
        >
          <div
            style={{
              background: '#111827',
              borderRadius: '16px',
              padding: '24px',
              maxWidth: '400px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <StrategyCardComponent card={selectedCard} />
            <div style={{ marginTop: '16px', textAlign: 'center' }}>
              <button
                onClick={() => setSelectedCard(null)}
                style={{
                  background: '#374151',
                  color: '#f3f4f6',
                  border: 'none',
                  padding: '8px 24px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
