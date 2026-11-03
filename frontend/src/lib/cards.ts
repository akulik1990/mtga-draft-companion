import type { Card, Rec } from './api';

export const COLORS = ['W', 'U', 'B', 'R', 'G'] as const;
export const COLOR_NAME: Record<string, string> = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };
export const COLOR_BG: Record<string, string> = {
	W: 'bg-amber-100 text-zinc-900', U: 'bg-sky-500 text-white', B: 'bg-zinc-700 text-white',
	R: 'bg-red-600 text-white', G: 'bg-green-600 text-white'
};

export const ALL = 'All Decks';
export const stats = (c: Card, lane = ALL) => c.deck_colors?.[lane];
export const gihwr = (c: Card, lane = ALL) => stats(c, lane)?.gihwr ?? 0;
export const isLand = (c: Card) => c.types?.includes('Land');

export function colorGroup(c: Card): string {
	const cs = c.colors ?? [];
	if (cs.length === 0) return isLand(c) ? 'Land' : 'Colorless';
	return cs.length === 1 ? cs[0] : 'Multi';
}

export const byName = (recs: Rec[] | undefined) =>
	new Map((recs ?? []).map((r) => [r.card_name, r]));

export const pct = (v: number | undefined | null, d = 1) =>
	v == null ? '-' : `${v.toFixed(d)}%`;

export const gradeTone = (g: string) =>
	g.startsWith('A') ? 'text-emerald-400' : g.startsWith('B') ? 'text-lime-300' :
	g.startsWith('C') ? 'text-zinc-300' : g === '-' ? 'text-zinc-600' : 'text-rose-400';

/** The set-rank marker: "#1".."#5", then "Top 10" / "Top 20" / "Top 30"; '' below that. */
export const rankMark = (r?: number | null) => (!r || r > 30 ? '' : r <= 5 ? `#${r}` : r <= 10 ? 'Top 10' : r <= 20 ? 'Top 20' : 'Top 30');
