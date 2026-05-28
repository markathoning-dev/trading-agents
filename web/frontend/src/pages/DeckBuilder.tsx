import React, { useEffect, useState } from 'react';
import { listCards, listDecks, createDeck, deleteDeck } from '../api/cards';
import { StrategyCard, Deck, RARITY_COLORS } from '../types/cards';
import { StrategyCardComponent } from '../components/StrategyCard';

export const DeckBuilder: React.FC = () => {
  const [cards, setCards] = useState<StrategyCard[]>([]);
  const [decks, setDecks] = useState<Deck[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Deck builder state
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
  const hasReward = rewardCards.length === 1;

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
    return <div style={{ color: '#9ca3af', padding: '20px' }}>Loading...</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1 style={{ color: '#f3f4f6', marginBottom: '20px' }}>Deck Builder</h1>

      {error && (
        <div style={{ background: '#7f1d1d', color: '#fecaca', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Left: Card Collection */}
        <div>
          <h2 style={{ color: '#d1d5db', marginBottom: '12px' }}>Available Cards</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '600px', overflowY: 'auto' }}>
            {cards.map((card) => (
              <div
                key={card.id}
                style={{ display: 'flex', alignItems: 'center', gap: '12px' }}
              >
                <StrategyCardComponent card={card} compact onClick={addCard} />
                <button
                  onClick={() => addCard(card)}
                  disabled={!!selectedCards.find((c) => c.id === card.id)}
                  style={{
                    background: selectedCards.find((c) => c.id === card.id) ? '#374151' : '#3b82f6',
                    color: '#fff',
                    border: 'none',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    cursor: selectedCards.find((c) => c.id === card.id) ? 'not-allowed' : 'pointer',
                    opacity: selectedCards.find((c) => c.id === card.id) ? 0.5 : 1,
                  }}
                >
                  Add
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Deck Builder */}
        <div>
          <h2 style={{ color: '#d1d5db', marginBottom: '12px' }}>Your Deck</h2>

          {/* Deck Info */}
          <div style={{ background: '#1f2937', padding: '16px', borderRadius: '8px', marginBottom: '16px' }}>
            <div style={{ marginBottom: '12px' }}>
              <input
                type="text"
                placeholder="Deck ID (e.g., my-deck)"
                value={deckId}
                onChange={(e) => setDeckId(e.target.value)}
                style={{
                  background: '#374151',
                  border: 'none',
                  color: '#f3f4f6',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  width: '100%',
                  marginBottom: '8px',
                }}
              />
              <input
                type="text"
                placeholder="Deck Name"
                value={deckName}
                onChange={(e) => setDeckName(e.target.value)}
                style={{
                  background: '#374151',
                  border: 'none',
                  color: '#f3f4f6',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  width: '100%',
                }}
              />
            </div>

            {/* Mana Budget */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
              <span style={{ color: '#9ca3af' }}>Mana Budget:</span>
              <input
                type="number"
                min={1}
                max={20}
                value={manaBudget}
                onChange={(e) => setManaBudget(Number(e.target.value))}
                style={{
                  background: '#374151',
                  border: 'none',
                  color: '#f3f4f6',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  width: '80px',
                }}
              />
              <span style={{ color: overBudget ? '#ef4444' : '#10b981' }}>
                {totalMana}/{manaBudget} mana
              </span>
            </div>

            {/* Validation */}
            {validationErrors.length > 0 && (
              <div style={{ color: '#ef4444', fontSize: '12px' }}>
                {validationErrors.map((err, i) => (
                  <div key={i}>⚠ {err}</div>
                ))}
              </div>
            )}
          </div>

          {/* Selected Cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
            {selectedCards.map((card) => (
              <div
                key={card.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  background: '#1f2937',
                  padding: '8px 12px',
                  borderRadius: '8px',
                }}
              >
                <div style={{ flex: 1 }}>
                  <span style={{ color: '#f3f4f6' }}>{card.name}</span>
                  <span style={{ color: RARITY_COLORS[card.rarity], marginLeft: '8px', fontSize: '12px' }}>
                    {card.rarity}
                  </span>
                </div>
                <span style={{ color: '#60a5fa' }}>⚡{card.mana_cost}</span>
                <button
                  onClick={() => removeCard(card.id)}
                  style={{
                    background: '#7f1d1d',
                    color: '#fecaca',
                    border: 'none',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            disabled={!isValid || !deckId || !deckName || saving}
            style={{
              background: isValid && deckId && deckName ? '#10b981' : '#374151',
              color: '#fff',
              border: 'none',
              padding: '12px 24px',
              borderRadius: '8px',
              cursor: isValid && deckId && deckName ? 'pointer' : 'not-allowed',
              width: '100%',
            }}
          >
            {saving ? 'Saving...' : 'Save Deck'}
          </button>

          {/* Existing Decks */}
          <h3 style={{ color: '#d1d5db', marginTop: '24px', marginBottom: '12px' }}>Saved Decks</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {decks.map((deck) => (
              <div
                key={deck.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: '#1f2937',
                  padding: '12px 16px',
                  borderRadius: '8px',
                }}
              >
                <div>
                  <span style={{ color: '#f3f4f6' }}>{deck.name}</span>
                  <span style={{ color: '#6b7280', marginLeft: '8px', fontSize: '12px' }}>
                    {deck.card_ids.length} cards
                  </span>
                </div>
                <button
                  onClick={() => handleDelete(deck.id)}
                  style={{
                    background: '#7f1d1d',
                    color: '#fecaca',
                    border: 'none',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
