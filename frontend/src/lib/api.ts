export type Stats = {
	gihwr: number; ohwr: number; gpwr: number; gnswr: number; gdwr: number;
	alsa: number; ata: number; iwd: number;
	samples: number; game_count: number; seen_count: number; pick_count: number; play_rate: number;
};
export type Card = {
	name: string; cmc: number; mana_cost: string; colors: string[]; types: string[]; subtypes: string[];
	rarity: string; image: string[]; tags: string[]; oracle_text?: string;
	deck_colors: Record<string, Stats>;
	grades: Record<string, Record<string, string>>;
	returnable_at?: number[];
};
export type Rec = {
	rank: number; card_name: string; base_win_rate: number; contextual_score: number; z_score: number;
	cast_probability: number; wheel_chance: number; functional_cmc: number; reasoning: string[];
	is_elite: boolean; archetype_fit: string; tags: string[];
};
export type AdvisorKey = 'upstream' | 'v2';
export type Mock = { kind: 'table' | '17lands'; id: number | string; set: string; done: boolean; pack_number: number | null; pick_number: number | null; picks: number; human: string | null };
export type Snapshot = {
	status: string;
	mock: Mock | null;
	event: { set: string; type: string; label: string; draft_id: string };
	pack_number: number; pick_number: number; cards_per_pick: number;
	dataset: string; archetype: string;
	color_ratings: Record<string, number>;
	signals: Record<string, number>;
	lanes: Record<string, number>;
	pack: Card[]; missing: Card[]; picked_from_pack: Card[]; pool: Card[];
	advisors: Partial<Record<AdvisorKey, Rec[]>>;
};
export type DatasetRow = {
	set: string; format: string; user_group: string; start_date: string; end_date: string;
	games: number; file: string; collected: string;
};
export type Datasets = { active: string; datasets: DatasetRow[] };
export type Settings = {
	arena_log_location: string; arena_log_pinned: boolean; auto_sync_datasets: boolean; active_log: string;
};
export type DeckCard = {
	name: string; count: number; cmc: number; mana_cost: string; types: string[]; colors: string[];
	image: string[]; rarity: string; gihwr: number | null;
};
export type Build = {
	label: string; variant: string; rating: number | null; record: string | null; colors: string[];
	breakdown: Record<string, unknown> | null; stats: Record<string, number> | null; note: string | null;
	deck: DeckCard[]; sideboard: DeckCard[]; arena_export: string;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
	const r = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init });
	if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
	return r.json();
}
const post = <T>(path: string, body?: unknown) =>
	call<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
	state: () => call<Snapshot>('/api/state'),
	rescan: () => post('/api/draft/rescan'),
	reset: () => post('/api/draft/reset'),
	datasets: () => call<Datasets>('/api/datasets'),
	selectDataset: (set: string, format: string, user_group = 'All') =>
		post<Datasets>('/api/datasets/select', { set, format, user_group }),
	syncDatasets: () => post<Datasets & { messages: string[] }>('/api/datasets/sync'),
	settings: () => call<Settings>('/api/settings'),
	updateSettings: (changes: Partial<Settings>) =>
		call<Settings>('/api/settings', { method: 'PUT', body: JSON.stringify(changes) }),
	buildDecks: () => post<Build[]>('/api/deck/build'),
	mockStart: (kind: 'random' | 'table' | '17lands' = 'random') => post<Mock>('/api/mock/start', { kind }),
	mockPick: (card_name: string) => post<Mock>('/api/mock/pick', { card_name }),
	mockStop: () => post('/api/mock/stop')
};
