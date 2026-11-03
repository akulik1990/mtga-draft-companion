<script lang="ts">
	import { untrack } from 'svelte';
	import { api, type DeckCard, type DeckScore, type ScoredCard } from './api';
	import { show, hide } from './hover.svelte';
	import Pips from './Pips.svelte';

	let { start, onclose }: { start: Record<string, number>; onclose: () => void } = $props();

	const BASICS: [string, string][] = [['W', 'Plains'], ['U', 'Island'], ['B', 'Swamp'], ['R', 'Mountain'], ['G', 'Forest']];
	let counts = $state<Record<string, number>>(untrack(() => ({ ...start })));  // a copy: edits never touch the build
	let s = $state<DeckScore | null>(null);
	let error = $state('');
	let copied = $state(false);
	let sortKey = $state<keyof ScoredCard>('value');
	let timer: ReturnType<typeof setTimeout> | undefined;
	let seq = 0;

	// Re-score on every change, a beat after the last click.
	$effect(() => {
		const snapshot = { ...counts };
		clearTimeout(timer);
		timer = setTimeout(async () => {
			const mine = ++seq;  // a slower earlier answer must not overwrite a newer one
			try { const r = await api.scoreDeck(snapshot); if (mine === seq) { s = r; error = ''; } } catch (e) { if (mine === seq) error = String(e); }
		}, 150);
	});

	const pool = $derived(new Map((s?.pool ?? []).map((c) => [c.name, c])));
	const add = (name: string, n = 1) => (counts = { ...counts, [name]: Math.max(0, (counts[name] ?? 0) + n) });
	const isLand = (c?: DeckCard) => !!c?.types?.includes('Land');
	const columns = $derived.by(() => {
		const cols: { label: string; cards: DeckCard[] }[] = [1, 2, 3, 4, 5, 6, 7].map((n) => ({ label: n === 7 ? '7+' : String(n), cards: [] }));
		const lands: DeckCard[] = [];
		for (const [name, k] of Object.entries(counts)) {
			const c = pool.get(name);
			if (!c) continue;
			for (let i = 0; i < Math.min(k, c.count); i++) (isLand(c) ? lands : cols[Math.min(6, Math.max(0, (c.cmc || 1) - 1))].cards).push(c);
		}
		return [...cols.filter((c) => c.cards.length), { label: 'Lands', cards: lands }];
	});
	const rows = $derived(
		[...(s?.cards ?? [])].sort((a, b) => {
			const x = a[sortKey] ?? -99, y = b[sortKey] ?? -99;
			return typeof x === 'string' ? String(x).localeCompare(String(y)) : (y as number) - (x as number);
		})
	);
	const fmt = (v: number | null | undefined, d = 1) => (v == null ? '–' : v.toFixed(d));
	const signed = (v: number | null | undefined) => (v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(1));
	async function copy() {
		if (!s) return;
		await navigator.clipboard.writeText(s.arena_export);
		copied = true; setTimeout(() => (copied = false), 1500);
	}
	const COLS: [keyof ScoredCard, string, string][] = [
		['bomb', '★', 'Rank among the set\'s 30 best cards by GIH WR'],
		['power', 'Pow', 'GIH WR in the card\'s best pair (its colours, ignoring this deck)'],
		['fit', 'Fit', 'GIH WR in this deck\'s main pair'],
		['mates', 'Mates', 'How much more often top players\' decks in this pair hold it with your other spells than chance (mean log lift)'],
		['inclusion', 'Incl', 'How often top players put a copy they hold into a deck of this pair, against the pair\'s average (log-odds)'],
		['value', 'Value', 'In the deck: score lost if swapped for the best sideboard spell. In the sideboard: score gained by the best swap. Score = predicted WR - shape penalty + synergy extras, as the builder judges'],
	];
</script>

<div class="mb-3 flex flex-wrap items-center gap-3">
	<span class="text-sm font-semibold">Your deck</span>
	{#if s}
		<Pips colors={s.main} />
		<span class="text-sm">{s.main}{s.splash ? ` + ${s.splash} splash` : ''}</span>
		<span class="font-mono text-sm {s.total === 40 ? 'text-zinc-300' : 'text-amber-400'}">{s.total} cards ({s.spells} spells, {s.lands} lands)</span>
		{#if s.deck_wr != null}<span class="font-mono text-emerald-300" title="Predicted game win rate for an average player (tools/deck_model.py)">{s.deck_wr.toFixed(1)}%</span>{/if}
		{#if s.shape_penalty}<span class="font-mono text-xs text-amber-400" title="Win-rate points the builder charges for straying from top players' deck shape in this pair">shape −{fmt(s.shape_penalty)}</span>{/if}
		<span class="text-sm text-amber-300" title="Cards from the set's 30 best by GIH WR">★ {s.bombs.in_deck} of {s.bombs.held} bombs in the deck</span>
	{/if}
	<button class="rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={copy} disabled={!s}>{copied ? 'copied' : 'Copy Arena export'}</button>
	<button class="rounded px-3 py-1 text-sm text-zinc-400 hover:bg-zinc-800" onclick={onclose}>Back to builds</button>
	{#if error}<span class="text-sm text-rose-400">{error}</span>{/if}
</div>

<div class="grid gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
	<section>
		<p class="mb-2 text-xs text-zinc-500">Click a card in the deck to take it out; add cards from the table. Basics with the buttons.</p>
		<div class="flex gap-2 overflow-x-auto pb-2">
			{#each columns as col (col.label)}
				<div class="w-32 shrink-0">
					<div class="mb-1 text-center font-mono text-xs text-zinc-500">{col.label} · {col.cards.length + (col.label === 'Lands' ? BASICS.reduce((n, [, b]) => n + (counts[b] ?? 0), 0) : 0)}</div>
					{#if col.label === 'Lands'}
						{#each BASICS as [colour, basic]}
							<div class="mb-1 flex items-center gap-1 rounded bg-zinc-900 px-2 py-1 text-sm">
								<Pips colors={colour} /><span class="w-5 text-center font-mono">{counts[basic] ?? 0}</span>
								<button class="rounded px-1 text-zinc-400 hover:bg-zinc-700" onclick={() => add(basic, -1)} aria-label="one fewer {basic}">−</button>
								<button class="rounded px-1 text-zinc-400 hover:bg-zinc-700" onclick={() => add(basic)} aria-label="one more {basic}">+</button>
							</div>
						{/each}
					{/if}
					<div class="relative" style="height: {col.cards.length ? `calc(${col.cards.length - 1} * 1.6rem + 11.2rem)` : '0'}">
						{#each col.cards as c, i}
							<button class="absolute block w-32" style="top: {i * 1.6}rem" onclick={() => { hide(); add(c.name, -1); }} aria-label="take out {c.name}">
								<img src={c.image?.[0]} alt={c.name} loading="lazy" class="w-32 rounded-lg shadow hover:z-10 hover:ring-2 hover:ring-rose-400"
									onmouseenter={(e) => show(c, e)} onmouseleave={hide} />
							</button>
						{/each}
					</div>
				</div>
			{/each}
		</div>

		<h3 class="mt-4 mb-1 text-sm font-semibold text-zinc-400">Pool</h3>
		<table class="w-full text-sm">
			<thead><tr class="text-left text-xs text-zinc-500">
				<th class="py-1"><button onclick={() => (sortKey = 'name')}>Card</button></th>
				<th class="text-center">Deck</th>
				{#each COLS as [k, label, tip]}<th class="text-right" title={tip}><button class={sortKey === k ? 'text-zinc-200' : ''} onclick={() => (sortKey = k)}>{label}</button></th>{/each}
				<th></th>
			</tr></thead>
			<tbody>
				{#each rows as r (r.name)}
					{@const c = pool.get(r.name)}
					<tr class="border-t border-zinc-800 {r.in_deck ? '' : 'text-zinc-400'}">
						<td class="py-0.5"><button class="flex items-center gap-2 text-left" onmouseenter={(e) => c && show(c, e)} onmouseleave={hide}>
							<span class="flex w-10 shrink-0"><Pips colors={c?.colors ?? []} /></span>{r.name}
							{#if r.status !== 'on'}<span class="text-xs {r.status === 'splash' ? 'text-sky-400' : 'text-rose-400'}">{r.status}</span>{/if}
						</button></td>
						<td class="text-center font-mono">{r.in_deck}/{r.held}</td>
						<td class="text-right font-mono text-amber-300">{r.bomb ?? ''}</td>
						<td class="text-right font-mono" title={r.power_lane}>{fmt(r.power)}</td>
						<td class="text-right font-mono">{fmt(r.fit)}</td>
						<td class="text-right font-mono">{fmt(r.mates, 2)}</td>
						<td class="text-right font-mono">{signed(r.inclusion)}</td>
						<td class="text-right font-mono {r.value == null ? '' : r.in_deck ? 'text-zinc-200' : r.value > 0 ? 'text-emerald-300' : ''}"
							title={r.swap_for ? (r.in_deck ? `best replacement: ${r.swap_for}` : `best swap: out ${r.swap_for}`) : ''}>{signed(r.value)}</td>
						<td class="text-right whitespace-nowrap">
							{#if r.in_deck}<button class="rounded px-1.5 text-zinc-400 hover:bg-zinc-700" onclick={() => add(r.name, -1)} aria-label="take out {r.name}">−</button>{/if}
							{#if r.in_deck < r.held}<button class="rounded px-1.5 text-zinc-400 hover:bg-zinc-700" onclick={() => add(r.name)} aria-label="add {r.name}">+</button>{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</section>

	<aside class="flex flex-col gap-4">
		{#if s}
			<section>
				<h3 class="mb-1 text-sm font-semibold text-zinc-400">Shape vs top players in {s.main}</h3>
				<table class="w-full font-mono text-sm"><tbody>
					{#each s.shape as row}
						{@const off = row.top != null ? row.deck - row.top : 0}
						<tr><td class="text-zinc-500">{row.name.replace('_', ' ')}</td>
							<td class="text-right">{row.deck}</td>
							<td class="text-right text-zinc-500">{row.top != null ? row.top.toFixed(1) : '–'}</td>
							<td class="w-12 text-right {Math.abs(off) >= 2 ? 'text-amber-400' : 'text-zinc-500'}">{row.top != null ? (off > 0 ? '+' : '') + off.toFixed(1) : ''}</td></tr>
					{/each}
				</tbody></table>
			</section>
			<section>
				<h3 class="mb-1 text-sm font-semibold text-zinc-400">Mana</h3>
				<table class="w-full font-mono text-sm"><tbody>
					{#each s.mana as m}
						<tr><td><Pips colors={m.colour} /></td>
							<td class="text-right">{m.sources} sources</td>
							<td class="text-right text-zinc-500">{m.spells} spells{m.max_pips > 1 ? `, up to ${m.max_pips} pips` : ''}</td></tr>
					{/each}
				</tbody></table>
			</section>
			<p class="text-xs text-zinc-500">Pow and Fit are 17Lands GIH WR. Value is in the builder's score units (win-rate points): the same model and shape rules the builder searches with, so a positive sideboard value means the builder's own measure prefers that swap.</p>
		{/if}
	</aside>
</div>
