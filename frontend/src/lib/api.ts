export type Stats = {
	gihwr: number; ohwr: number; gpwr: number; gnswr: number; gdwr: number;
	alsa: number; ata: number; iwd: number;
	samples: number; game_count: number; seen_count: number; pick_count: number; play_rate: number;
};
export type Card = {
	name: string; cmc: number; mana_cost: string; colors: string[]; types: string[]; subtypes: string[];
	rarity: string; image: string[]; tags: string[]; oracle_text?: string;
	set_rank?: number | null;
	deck_colors: Record<string, Stats>;
	grades: Record<string, Record<string, string>>;
	tiers?: Record<string, string>;
	returnable_at?: number[];
};
export type Rec = {
	rank: number; card_name: string; base_win_rate?: number; contextual_score: number; z_score: number;
	cast_probability?: number; wheel_chance: number; functional_cmc?: number; reasoning: string[];
	is_elite: boolean; archetype_fit?: string; tags: string[];
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
	signals_v2?: Record<string, number>;
	v2_split?: V2Split | null;
	signals_v2_sides?: { active?: string; right?: Record<string, number>; left?: Record<string, number> };
	pack: Card[]; missing: Card[]; picked_from_pack: Card[]; pool: Card[];
	advisors: Partial<Record<AdvisorKey, Rec[]>>;
	deck_filter: string;
	deck_filter_options: Record<string, string>; // label -> key
	result_format: ResultFormat;
	metrics: Record<string, Record<string, [number, number]>>; // lane -> field -> [mean, std]
	grade_thresholds: [string, number][];
	tiers: { key: string; label: string }[];
};
export type V2Split = {
	power: { card: string; score: number }[];
	fit: { card: string; score: number }[];
	branch: { card: string; over: string; gap: number; threshold: number; stage: string; in_colour_strength: number; take: boolean } | null;
};
export type ResultFormat = 'Percentage' | 'Rating' | 'Grade';
export type EngineSwitch = { on: boolean; label: string; help: string };
export type TierFile = { set: string; label: string; date: string; file: string };
export type DatasetRow = {
	set: string; format: string; user_group: string; start_date: string; end_date: string;
	games: number; file: string; collected: string;
};
export type Datasets = { active: string; datasets: DatasetRow[] };
export type Settings = {
	arena_log_location: string; arena_log_pinned: boolean; auto_sync_datasets: boolean; active_log: string;
	deck_filter: string; result_format: ResultFormat; result_formats: ResultFormat[];
};
export type DeckCard = {
	name: string; count: number; cmc: number; mana_cost: string; types: string[]; colors: string[];
	image: string[]; rarity: string; gihwr: number | null;
};
export type Build = {
	label: string; variant: string; rating: number | null; record: string | null; colors: string[];
	breakdown: Record<string, unknown> | null; stats: Record<string, number> | null; note: string | null;
	deck: DeckCard[]; sideboard: DeckCard[]; arena_export: string;
	main: string; splash: string; splash_cards: number; deck_wr: number | null; model_wr?: number | null; creatures?: number; realistic?: boolean;
};
export type ScoredCard = {
	name: string; in_deck: number; held: number; land: boolean; status: 'on' | 'splash' | 'off';
	power: number; power_lane: string; fit: number | null; mates: number | null; inclusion: number | null;
	bomb: number | null; value?: number; swap_for?: string | null;
};
export type DeckScore = {
	main: string; splash: string; total: number; spells: number; lands: number;
	deck_wr: number | null; shape_penalty: number | null; extras: number | null; objective: number | null;
	shape: { name: string; deck: number; top: number | null }[];
	mana: { colour: string; sources: number; spells: number; max_pips: number }[];
	cards: ScoredCard[]; bombs: { held: number; in_deck: number };
	pool: DeckCard[]; arena_export: string;
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
	scoreDeck: (deck: Record<string, number>) => post<DeckScore>('/api/deck/score', { deck }),
	mockStart: (kind: 'random' | 'table' | '17lands' = 'random') => post<Mock>('/api/mock/start', { kind }),
	mockPick: (card_name: string) => post<Mock>('/api/mock/pick', { card_name }),
	mockStop: () => post('/api/mock/stop'),
	engine: () => call<Record<string, EngineSwitch>>('/api/engine'),
	setEngine: (changes: Record<string, boolean>) => call<Record<string, EngineSwitch>>('/api/engine', { method: 'PUT', body: JSON.stringify(changes) }),
	tiers: () => call<TierFile[]>('/api/tiers'),
	importTier: (url: string, label = '') => post<TierFile[]>('/api/tiers', { url, label }),
	deleteTier: (file: string) => call<TierFile[]>(`/api/tiers/${encodeURIComponent(file)}`, { method: 'DELETE' }),
	cards: (q: string) => call<Card[]>(`/api/cards?q=${encodeURIComponent(q)}`),
	isStatic,
	videos: async () => (await isStatic() ? staticVideos() : call<VideoRow[]>('/api/videos')),
	video: async (id: string) => (await isStatic() ? staticVideo(id) : call<VideoData>(`/api/videos/${id}`)),
	videoEngine: (id: string, commit: string) => call<VideoEngine>(`/api/videos/${id}/engine/${commit}`).catch(() => getJson<VideoEngine>(`${id}/engine/${commit}.json`)),
	setIdea: (id: string, idea: string, body: IdeaStatus) =>
		call<IdeaStatus>(`/api/videos/${id}/ideas/${encodeURIComponent(idea)}`, { method: 'PUT', body: JSON.stringify(body) })
};

// Static site (GitHub Pages, tools/video_site.py): no API, so the video records are read as files next to the page
// (videos/index.html -> ../data/videos/...). The checklist is then read-only.
let staticMode: Promise<boolean> | undefined;
function isStatic() {
	return (staticMode ??= fetch('/api/health').then((r) => !r.ok, () => true).catch(() => true));
}
async function getJson<T>(path: string, fallback?: T): Promise<T> {
	const r = await fetch(new URL(`../data/videos/${path}`, location.href));
	if (r.ok) return r.json();
	if (fallback !== undefined) return fallback;
	throw new Error(`${r.status} ${path}`);
}
async function staticVideos(): Promise<VideoRow[]> {
	const rows = await getJson<VideoRow[]>('index.json');
	return Promise.all(rows.map(async (v) => {
		const e = await getJson<Partial<VideoEngine>>(`${v.id}/engine/latest.json`, {});
		return { ...v, top1: e.top1 ?? null, top3: e.top3 ?? null, n: e.n ?? null, commit: e.commit ?? null };
	}));
}
async function staticVideo(id: string): Promise<VideoData> {
	const video = (await getJson<VideoRow[]>('index.json')).find((v) => v.id === id);
	if (!video) throw new Error(`No video ${id}`);
	const [picks, deck, ideas, status, engine, commits] = await Promise.all([
		getJson<VideoData['picks']>(`${id}/picks.json`), getJson<VideoData['deck']>(`${id}/deck.json`, null),
		getJson<Idea[]>(`${id}/ideas.json`, []), getJson<VideoData['status']>(`${id}/ideas-status.json`, {}),
		getJson<VideoEngine | null>(`${id}/engine/latest.json`, null), getJson<string[]>(`${id}/engine/commits.json`, [])
	]);
	const list = Array.isArray(ideas) ? ideas : ((ideas as { ideas?: Idea[] } | null)?.ideas ?? []); // ideas.json may carry its metadata
	return { video, picks, deck, ideas: list, status, engine, commits, readonly: true };
}

export const IDEA_STATUSES = ['untested', 'testing', 'confirmed', 'rejected', 'shipped'] as const;
export type IdeaStatus = { status: string; result: string; doc: string };
export type VideoRow = { id: string; title: string; player: string; set: string; url: string; added: string; top1: number | null; top3: number | null; n: number | null; commit: string | null };
export type VideoPick = { pack: number; pick: number; time: string; taken: string; reasoning: string; pack_cards: string[]; plan?: string };
export type Idea = { id: string; time: string | number; phase: string; claim: string; quote?: string; cards?: string[]; kind?: string; engine_area?: string; test_idea?: string };
export type EngineDeck = { main: string | null; splash: string | null; deck_wr: number | null; label: string | null; cards: { name: string; count: number }[] };
export type EnginePick = {
	pack: number; pick: number; taken: string; top3: { name: string; score: number }[]; his_score: number | null; his_rank: number | null;
	reasons_top: string[]; reasons_his: string[]; lanes_after: Record<string, number>;
};
export type VideoEngine = { commit: string; date: string; top1: number; top3: number; n: number; picks: EnginePick[]; builder: EngineDeck[]; his_deck: EngineDeck | null };
export type VideoData = {
	video: VideoRow;
	picks: { picks: VideoPick[]; colour_path: { after: string; main: string; quote?: string }[] };
	deck: { main: Record<string, number>; lands: Record<string, number>; sideboard?: Record<string, number> } | null;
	ideas: Idea[]; status: Record<string, IdeaStatus>; engine: VideoEngine | null; commits: string[]; readonly?: boolean;
};
