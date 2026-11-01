// Upstream's three result formats (card_logic.format_win_rate), computed in the browser so any
// lane and field can be shown without a server round trip.
import type { Card, Rec, ResultFormat, Snapshot } from './api';
import { ALL } from './cards';

export const WR_FIELDS = ['gihwr', 'ohwr', 'gpwr', 'gdwr', 'gnswr'] as const;
export const FIELD_LABEL: Record<string, string> = {
	gihwr: 'GIH', ohwr: 'OH', gpwr: 'GP', gdwr: 'GD', gnswr: 'GNS', iwd: 'IWD', alsa: 'ALSA', ata: 'ATA'
};
export const FIELD_TITLE: Record<string, string> = {
	gihwr: 'Games in hand win rate', ohwr: 'Opening hand win rate', gpwr: 'Games played win rate',
	gdwr: 'Games drawn win rate', gnswr: 'Games not seen win rate', iwd: 'Improvement when drawn',
	alsa: 'Average last seen at', ata: 'Average taken at'
};

export function raw(card: Card, field: string, lane = ALL): number {
	return Number((card.deck_colors?.[lane] as Record<string, number> | undefined)?.[field] ?? 0);
}

/** The value as upstream would print it: "-" when there is no data. */
export function fmt(snap: Snapshot, card: Card, field: string, lane = ALL, format: ResultFormat = snap.result_format): string {
	const v = raw(card, field, lane);
	if (!v) return '-';
	if (!(WR_FIELDS as readonly string[]).includes(field)) return field === 'iwd' ? v.toFixed(1) : v.toFixed(2);
	if (format === 'Percentage') return v.toFixed(1);
	const [mean, std] = snap.metrics?.[lane]?.[field] ?? snap.metrics?.[ALL]?.[field] ?? [0, 0];
	if (!std) return v.toFixed(1);
	const z = (v - mean) / std;
	if (format === 'Grade') {
		for (const [g, limit] of snap.grade_thresholds ?? []) if (z >= limit) return g;
		return 'F';
	}
	const upper = mean + 2 * std, lower = mean - 1.67 * std;
	return Math.max(0, Math.min(5, ((v - lower) / (upper - lower)) * 5)).toFixed(1);
}

/** Colour for a formatted value, by its z-score whatever the format. */
export function tone(snap: Snapshot, card: Card, field: string, lane = ALL): string {
	const v = raw(card, field, lane);
	if (!v || !(WR_FIELDS as readonly string[]).includes(field)) return 'text-zinc-300';
	const [mean, std] = snap.metrics?.[lane]?.[field] ?? [0, 0];
	if (!std) return 'text-zinc-300';
	const z = (v - mean) / std;
	return z >= 1.33 ? 'text-emerald-400' : z >= 0.33 ? 'text-lime-300' : z >= -0.67 ? 'text-zinc-300' : 'text-rose-400';
}

export const COLOR_STRIPE: Record<string, string> = {
	W: 'border-l-amber-100', U: 'border-l-sky-500', B: 'border-l-zinc-500', R: 'border-l-red-600', G: 'border-l-green-600'
};
export function stripe(card: Card): string {
	const cs = card.colors ?? [];
	return cs.length === 0 ? 'border-l-zinc-700' : cs.length > 1 ? 'border-l-yellow-500' : COLOR_STRIPE[cs[0]] ?? 'border-l-zinc-700';
}

const TAG_ICON: Record<string, string> = {
	removal: '🎯', card_advantage: '🃏', evasion: '🪽', fixing_ramp: '💎', lifegain: '❤', combat_trick: '⚔',
	token_maker: '👥', counterspell: '🚫', protection: '🛡', mana_sink: '🔋'
};
export const tagIcons = (tags: string[] | undefined) => (tags ?? []).map((t) => TAG_ICON[t] ?? '').filter(Boolean).join('');

export type Column = { key: string; label: string; title: string; sort: (c: Card) => number; show: (c: Card) => string; cls?: (c: Card) => string; kind?: 'colors' };

/** Every column upstream's pack table offers, plus both advisors' scores. */
export function columns(snap: Snapshot, up: Map<string, Rec>, v2: Map<string, Rec>): Column[] {
	const lane = snap.archetype || ALL;
	const out: Column[] = [];
	for (const f of [...WR_FIELDS, 'iwd', 'alsa', 'ata']) {
		const asc = f === 'alsa' || f === 'ata';
		out.push({
			key: f, label: FIELD_LABEL[f], title: `${FIELD_TITLE[f]} (${lane})`,
			sort: (c) => (asc ? raw(c, f, lane) || 99 : -raw(c, f, lane)),
			show: (c) => fmt(snap, c, f, lane), cls: (c) => tone(snap, c, f, lane)
		});
	}
	out.push(
		{ key: 'wheel', label: 'WHEEL', title: 'Chance the card comes back', sort: (c) => -(up.get(c.name)?.wheel_chance ?? 0), show: (c) => { const w = up.get(c.name)?.wheel_chance; return w ? `${w.toFixed(0)}%` : '-'; } },
		{ key: 'colors', label: 'COLORS', title: 'Card colours', sort: (c) => (c.colors ?? []).length, show: (c) => (c.colors ?? []).join(''), kind: 'colors' },
		{ key: 'tags', label: 'TAGS', title: 'Card roles', sort: (c) => -(c.tags ?? []).length, show: (c) => tagIcons(c.tags) || '-' },
		{ key: 'upstream', label: 'UP', title: "Upstream advisor's score", sort: (c) => up.get(c.name)?.rank ?? 99, show: (c) => { const r = up.get(c.name); return r ? r.contextual_score.toFixed(0) : '-'; }, cls: () => 'text-sky-300' },
		{ key: 'v2', label: 'V2', title: "Our advisor's score", sort: (c) => v2.get(c.name)?.rank ?? 99, show: (c) => { const r = v2.get(c.name); return r ? r.contextual_score.toFixed(1) : '-'; }, cls: () => 'text-violet-300' }
	);
	for (const t of snap.tiers ?? []) {
		out.push({ key: `tier:${t.key}`, label: t.label.replace(/^TIER: /, '').slice(0, 14), title: t.label, sort: (c) => tierRank(c.tiers?.[t.key]), show: (c) => c.tiers?.[t.key] ?? '-' });
	}
	return out;
}

const TIER_ORDER = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F'];
const tierRank = (g?: string) => { const i = TIER_ORDER.indexOf((g ?? '').trim()); return i < 0 ? 99 : i; };

export const DEFAULT_COLUMNS = ['gihwr', 'ohwr', 'gpwr', 'iwd', 'ata', 'wheel', 'colors', 'tags', 'upstream', 'v2'];

export function loadPref<T>(key: string, fallback: T): T {
	try { const v = localStorage.getItem(key); return v ? (JSON.parse(v) as T) : fallback; } catch { return fallback; }
}
export function savePref(key: string, value: unknown) {
	try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window */ }
}
