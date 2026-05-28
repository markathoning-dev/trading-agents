import { get, postForm } from './client';
import { StrategyCard, Deck } from '../types/cards';

export async function listCards(): Promise<StrategyCard[]> {
  return get<StrategyCard[]>('/cards');
}

export async function getCard(cardId: string): Promise<StrategyCard> {
  return get<StrategyCard>(`/cards/${cardId}`);
}

export async function listDecks(): Promise<Deck[]> {
  return get<Deck[]>('/decks');
}

export async function getDeck(deckId: string): Promise<Deck> {
  return get<Deck>(`/decks/${deckId}`);
}

export async function createDeck(
  deckId: string,
  name: string,
  cardIds: string[],
  manaBudget: number
): Promise<Deck> {
  return postForm<Deck>('/decks', {
    deck_id: deckId,
    name,
    card_ids: cardIds.join(','),
    mana_budget: manaBudget,
  });
}

export async function deleteDeck(deckId: string): Promise<void> {
  const res = await fetch(`/api/decks/${deckId}`, { method: 'DELETE' });
  if (!res.ok) {
    throw new Error(`Failed to delete deck: ${res.statusText}`);
  }
}

export async function startBacktestWithDeck(
  modelName: string,
  symbol: string,
  maxSteps: number,
  deckId?: string
): Promise<{ run_id: number; status: string }> {
  const data: Record<string, string | number> = {
    model_name: modelName,
    symbol,
    max_steps: maxSteps,
  };
  if (deckId) {
    data.deck_id = deckId;
  }
  return postForm('/backtests/new', data);
}
