import React, { useEffect, useState } from 'react';
import { listCards, listDecks, createDeck, deleteDeck } from '../api/cards';
import { StrategyCard, Deck, RARITY_COLORS } from '../types/cards';
import { CmdInput } from '../components/CmdInput';
import { Btn } from '../components/Btn';

export const DeckBuilder: React.FC = () => {
  const [cards, setCards] = useState<StrategyCard[]>([]);
  const [decks, setDecks] = useState<Deck[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [deckName, setDeckName] = useState('');
  const [deckId, setDeckId] = useState('');
  const [selectedCards, setSelectedCards] = useState<StrategyCard[]>([]);
  const [manaBudget, setManaBudget] = useState(10);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    Promise.all([listCards(), listDecks()])
      .then(([cardsData, decksData]) => {
        setCards(cardsData);
        setDecks(decksData);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const totalMana = selectedCards.reduce((sum, card) => sum + card.mana_cost, 0);
  const overBudget = totalMana > manaBudget;
  const rewardCards = selectedCards.filter((c) => c.reward_type);

  const validationErrors: string[] = [];
  if (overBudget) validationErrors.push(`Over mana budget: ${totalMana}/${manaBudget}`);
  if (rewardCards.length === 0) validationErrors.push('Need exactly 1 reward card');
  if (rewardCards.length > 1) validationErrors.push(`Too many reward cards: ${rewardCards.length}`);
  if (selectedCards.length === 0) validationErrors.push('Deck is empty');

  const isValid = validationErrors.length === 0;

  const addCard = (card: StrategyCard) => {
    if (!selectedCards.find((c) => c.id === card.id)) {
      setSelectedCards([...selectedCards, card]);
    }
  };

  const removeCard = (cardId: string) => {
    setSelectedCards(selectedCards.filter((c) => c.id !== cardId));
  };

  const handleSave = async () => {
    if (!isValid || !deckId || !deckName) return;
    setSaving(true);
    try {
      await createDeck(deckId, deckName, selectedCards.map((c) => c.id), manaBudget);
      const updatedDecks = await listDecks();
      setDecks(updatedDecks);
      setDeckId('');
      setDeckName('');
      setSelectedCards([]);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteDeck(id);
      setDecks(decks.filter((d) => d.id !== id));
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <div className="py-4">
        <p className="text-text-muted animate-flicker">Loading...</p>
      </div>
    );
  }

  return (
    <div className="py-4">
      <h1 className="text-sm text-terminal-green tracking-wider mb-6">DECK BUILDER</h1>

      {error && (
        <div className="bg-terminal-red/10 border border-terminal-red/30 text-terminal-red p-3 rounded font-mono text-[12px] mb-4">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Card Pool */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-3">
            Available Cards
          </div>
          <div className="space-y-1 max-h-[600px] overflow-y-auto border border-screen-border rounded bg-screen-elevated p-2">
            {cards.map((card) => {
              const inDeck = !!selectedCards.find((c) => c.id === card.id);
              const color = RARITY_COLORS[card.rarity];
              return (
                <div
                  key={card.id}
                  className="flex items-center justify-between p-2 hover:bg-[#0d0d0d] transition-colors rounded"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="w-1.5 h-1.5 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-text-primary text-[13px] truncate">{card.name}</span>
                    <span className="text-terminal-cyan text-[12px] shrink-0">
                      {'\u26A1'.repeat(card.mana_cost)}
                    </span>
                  </div>
                  <button
                    onClick={() => addCard(card)}
                    disabled={inDeck}
                    className={`font-mono text-[11px] px-2 py-0.5 border transition-colors shrink-0 ml-2 ${
                      inDeck
                        ? 'border-screen-border text-text-dim cursor-not-allowed'
                        : 'border-terminal-green text-terminal-green hover:bg-terminal-green/10 cursor-pointer'
                    }`}
                  >
                    {inDeck ? 'Added' : '+'}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Active Deck */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-3">
            Active Deck
          </div>

          <div className="bg-screen-elevated border border-screen-border rounded p-4 space-y-4">
            <CmdInput
              label="deck_id"
              value={deckId}
              onChange={(e) => setDeckId(e.target.value)}
              placeholder="my-deck"
            />
            <CmdInput
              label="name"
              value={deckName}
              onChange={(e) => setDeckName(e.target.value)}
              placeholder="Deck Name"
            />

            <CmdInput
              label="mana"
              value={manaBudget}
              onChange={(e) => setManaBudget(Number(e.target.value))}
              type="number"
            />
            <div className="text-[13px] font-mono">
              <span className={overBudget ? 'text-terminal-red' : 'text-terminal-green'}>
                {totalMana}/{manaBudget} mana
              </span>
            </div>

            {validationErrors.length > 0 && (
              <div className="text-terminal-red text-[12px] space-y-0.5 font-mono">
                {validationErrors.map((err, i) => (
                  <div key={i}>{'\u26A0'} {err}</div>
                ))}
              </div>
            )}

            <div className="space-y-1">
              {selectedCards.map((card) => {
                const color = RARITY_COLORS[card.rarity];
                return (
                  <div
                    key={card.id}
                    className="flex items-center justify-between bg-screen-bg border border-screen-border rounded p-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className="w-1.5 h-1.5 rounded-full shrink-0"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-text-primary text-[13px] truncate">{card.name}</span>
                      <span className="text-terminal-cyan text-[12px] shrink-0">
                        {'\u26A1'.repeat(card.mana_cost)}
                      </span>
                    </div>
                    <button
                      onClick={() => removeCard(card.id)}
                      className="text-terminal-red hover:bg-terminal-red/10 px-1 cursor-pointer shrink-0 ml-2"
                    >
                      {'\u2715'}
                    </button>
                  </div>
                );
              })}
            </div>

            <Btn onClick={handleSave} disabled={!isValid || !deckId || !deckName} loading={saving}>
              Save Deck
            </Btn>
          </div>

          {decks.length > 0 && (
            <div className="mt-6">
              <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-3">
                Saved Decks
              </div>
              <div className="space-y-1">
                {decks.map((deck) => (
                  <div
                    key={deck.id}
                    className="flex items-center justify-between bg-screen-elevated border border-screen-border rounded p-3"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-text-primary text-[13px]">{deck.name}</span>
                      <span className="text-text-dim text-[11px]">
                        {deck.card_ids.length} cards
                      </span>
                    </div>
                    <button
                      onClick={() => handleDelete(deck.id)}
                      className="text-terminal-red text-[12px] hover:bg-terminal-red/10 px-2 py-0.5 cursor-pointer font-mono"
                    >
                      Delete
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};