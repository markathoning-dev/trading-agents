import React, { useEffect, useState } from 'react';
import { listCards } from '../api/cards';
import { StrategyCard, RARITY_COLORS } from '../types/cards';
import { StrategyCardComponent } from '../components/StrategyCard';
import { TerminalTable } from '../components/TerminalTable';
import { Btn } from '../components/Btn';
import { Modal } from '../components/Modal';

type ViewMode = 'table' | 'grid';

export const CardCollection: React.FC = () => {
  const [cards, setCards] = useState<StrategyCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [view, setView] = useState<ViewMode>('table');
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

  const FILTERS = ['all', 'common', 'rare', 'epic', 'legendary'];

  if (loading) {
    return (
      <div className="py-4">
        <p className="text-text-muted animate-flicker">Loading cards...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-4">
        <div className="bg-terminal-red/10 border border-terminal-red/30 text-terminal-red p-4 rounded font-mono text-[13px]">
          Error: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="py-4">
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <h1 className="text-sm text-terminal-green tracking-wider">
          STRATEGY CARDS <span className="text-text-dim text-[11px]">[{cards.length} loaded]</span>
        </h1>
        <Btn variant="ghost" onClick={() => setView(view === 'table' ? 'grid' : 'table')}>
          {view === 'table' ? 'Grid View' : 'Table View'}
        </Btn>
      </div>

      <div className="flex flex-wrap gap-2 mb-6">
        {FILTERS.map((rarity) => (
          <button
            key={rarity}
            onClick={() => setFilter(rarity)}
            className={`font-mono text-[12px] px-3 py-1 border cursor-pointer transition-colors capitalize ${
              filter === rarity
                ? 'border-terminal-green text-terminal-green bg-terminal-green/10'
                : 'border-screen-border text-text-muted hover:border-screen-border-hi hover:text-text-primary'
            }`}
          >
            {rarity} ({rarity === 'all' ? cards.length : rarityCounts[rarity] || 0})
          </button>
        ))}
      </div>

      {view === 'table' ? (
        <TerminalTable<StrategyCard>
          columns={[
            {
              key: 'name',
              label: 'Card',
              render: (card) => (
                <button
                  onClick={() => setSelectedCard(card)}
                  className="text-terminal-cyan hover:underline text-left cursor-pointer"
                >
                  {card.name}
                </button>
              ),
            },
            {
              key: 'rarity',
              label: 'Rarity',
              render: (card) => {
                const color = RARITY_COLORS[card.rarity];
                return (
                  <span
                    className="text-[11px] uppercase tracking-wider font-bold px-2 py-0.5 rounded"
                    style={{ backgroundColor: `${color}20`, color, border: `1px solid ${color}40` }}
                  >
                    {card.rarity}
                  </span>
                );
              },
            },
            {
              key: 'mana_cost',
              label: 'Mana',
              render: (card) => (
                <span className="text-terminal-cyan tabular-nums">
                  {'\u26A1'.repeat(card.mana_cost)}
                  <span className="text-text-dim">
                    {'\u26A1'.repeat(Math.max(0, 5 - card.mana_cost))}
                  </span>
                </span>
              ),
            },
            {
              key: 'reward_type',
              label: 'Type',
              render: (card) => (
                <span className={card.reward_type ? 'text-terminal-cyan' : 'text-text-dim'}>
                  {card.reward_type || '\u2014'}
                </span>
              ),
            },
            {
              key: 'nodes',
              label: 'Nodes',
              render: (card) => (
                <span className="text-text-muted text-[11px]">{card.nodes.length}</span>
              ),
            },
          ]}
          rows={filteredCards}
          sortable
          defaultSortKey="rarity"
          defaultSortDir="desc"
          emptyMessage="No cards match filter."
          expandable={{
            render: (card) => (
              <div className="space-y-2 py-2">
                <p className="text-text-primary text-[13px]">{card.description}</p>
                <p className="text-text-dim text-[12px] italic">"{card.flavor_text}"</p>
                <div className="grid grid-cols-2 gap-2 mt-3">
                  {(['risk_tolerance', 'volatility_preference', 'drawdown_penalty', 'trade_frequency'] as const).map((stat) => (
                    <div key={stat} className="flex items-center gap-2">
                      <span className="text-[11px] text-text-muted w-28">
                        {stat.replace(/_/g, ' ')}
                      </span>
                      <div className="flex-1 h-2 bg-screen-bg border border-screen-border rounded overflow-hidden">
                        <div
                          className="h-full bg-terminal-green/60"
                          style={{ width: `${card.stats[stat] * 100}%` }}
                        />
                      </div>
                      <span className="text-[11px] text-text-dim tabular-nums w-8 text-right">
                        {card.stats[stat].toFixed(1)}
                      </span>
                    </div>
                  ))}
                </div>
                {card.nodes.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {card.nodes.map((node) => (
                      <span key={node} className="bg-screen-bg border border-screen-border px-2 py-0.5 text-[10px] text-text-muted font-mono">
                        {node}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ),
          }}
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCards.map((card) => (
            <StrategyCardComponent
              key={card.id}
              card={card}
              onClick={setSelectedCard}
              selected={selectedCard?.id === card.id}
            />
          ))}
        </div>
      )}

      <Modal
        open={!!selectedCard}
        onClose={() => setSelectedCard(null)}
        title={selectedCard ? `CARD DETAIL: ${selectedCard.name.toUpperCase()}` : ''}
      >
        {selectedCard && (
          <div className="space-y-3">
            <StrategyCardComponent card={selectedCard} />
            <div className="flex justify-end gap-2 mt-4">
              <Btn variant="ghost" onClick={() => setSelectedCard(null)}>
                Close
              </Btn>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};