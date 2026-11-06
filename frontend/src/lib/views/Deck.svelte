<script lang="ts">
	import { api, type Build, type DeckCard } from '../api';
	import { pct } from '../cards';
	import { show, hide } from '../hover.svelte';
	import Pips from '../Pips.svelte';
	import DeckEditor from '../DeckEditor.svelte';

	let builds = $state<Build[]>([]);
	let active = $state(0);
	let busy = $state(false);
	let error = $state('');
	let copied = $state(false);
	let ran = $state(false);
	let editing = $state<Record<string, number> | null>(null);  // your own deck: {name: copies}, from a build's copy or empty
	const copyOf = (x: Build) => Object.fromEntries(x.deck.map((c) => [c.name, c.count]));

	async function build() {
		busy = true; error = '';
		try { builds = (await api.buildDecks()).sort((a, b) => Number(b.realistic ?? true) - Number(a.realistic ?? true) || (b.deck_wr ?? 0) - (a.deck_wr ?? 0)); active = 0; ran = true; } catch (e) { error = String(e); } finally { busy = false; }
	}
	const b = $derived(builds[active]);
	const count = (cs: DeckCard[]) => cs.reduce((n, c) => n + c.count, 0);
	const isLand = (c: DeckCard) => c.types?.includes('Land');
	// Spells in columns by mana value (1..6, 7+), lands in their own column; one image per copy.
	const columns = $derived.by(() => {
		if (!b) return [];
		const cols: { label: string; cards: DeckCard[] }[] = [1, 2, 3, 4, 5, 6, 7].map((n) => ({ label: n === 7 ? '7+' : String(n), cards: [] }));
		const lands: DeckCard[] = [];
		for (const c of b.deck) if (c.image?.[0]) for (let i = 0; i < c.count; i++) (isLand(c) ? lands : cols[Math.min(6, Math.max(0, (c.cmc || 1) - 1))].cards).push(c);
		return [...cols.filter((c) => c.cards.length), { label: 'Lands', cards: lands }];
	});
	const BASIC: Record<string, string> = { Plains: 'W', Island: 'U', Swamp: 'B', Mountain: 'R', Forest: 'G' };
	const basics = $derived((b?.deck ?? []).filter((c) => !c.image?.[0]));
	const label = (x: Build) => (x.main ? `${x.main}${x.splash ? ` + ${x.splash} splash` : ''}${x.variant && x.note?.startsWith('companion') ? ` (${x.variant})` : ''}` : x.label);
	const wr = (x: Build) => x.model_wr ?? x.deck_wr;
	const record = (x: Build) => x.record ?? x.label.match(/\[Est: ([^\]]+)\]/)?.[1] ?? '';
	async function copy() {
		await navigator.clipboard.writeText(b.arena_export);
		copied = true; setTimeout(() => (copied = false), 1500);
	}
</script>

{#if editing}
	<DeckEditor start={editing} onclose={() => (editing = null)} />
{:else}
<div class="mb-3 flex flex-wrap items-center gap-2">
	{#if b}<button class="rounded bg-sky-800 px-3 py-1 text-sm hover:bg-sky-700" onclick={() => (editing = copyOf(b))} title="Copy this build into the editor and change it">Edit a copy</button>{/if}
	<button class="rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={() => (editing = {})} title="Build your own deck from the pool">Start empty</button>
	<button class="rounded bg-emerald-700 px-3 py-1 text-sm hover:bg-emerald-600 disabled:opacity-50" onclick={build} disabled={busy}>
		{busy ? 'building…' : 'Build decks'}
	</button>
	{#each builds as x, i}
		<button class="flex items-center gap-1.5 rounded px-2 py-1 text-sm {i === active ? 'bg-zinc-700' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'}" onclick={() => (active = i)} title={x.label}>
			<Pips colors={x.main || x.colors} />
			<span>{label(x)}</span>
			{#if wr(x) != null}<span class="font-mono {i === 0 ? 'text-emerald-300' : 'text-zinc-300'}" title="Predicted game win rate for an average player, from 17Lands games (tools/deck_model.py)">{wr(x)!.toFixed(1)}%</span>{/if}
			{#if x.realistic === false}<span class="text-amber-400" title="{x.creatures} creatures: fewer than real decks run, so the predicted win rate is outside the model's evidence">{x.creatures} creatures ⚠</span>{/if}
			<span class="font-mono text-xs text-zinc-500" title="upstream's estimate and Power rating">{record(x)}{x.rating != null ? ` · ${x.rating.toFixed(0)}` : ''}</span>
		</button>
	{/each}
	{#if error}<span class="text-sm text-rose-400">{error}</span>{/if}
</div>

{#if b}
	<div class="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
		<section>
			<h3 class="mb-2 text-sm font-semibold text-zinc-400">
				Deck ({count(b.deck)}) · {label(b)}{wr(b) != null ? ` · predicted win rate ${wr(b)!.toFixed(1)}%` : ''}{b.splash ? ` (${b.splash_cards} card${b.splash_cards === 1 ? '' : 's'})` : ''} · upstream: {b.label}
			</h3>
			<div class="flex gap-2 overflow-x-auto pb-2">
				{#each columns as col (col.label)}
					<div class="w-32 shrink-0">
						<div class="mb-1 text-center font-mono text-xs text-zinc-500">{col.label} · {col.cards.length + (col.label === 'Lands' ? basics.reduce((n, c) => n + c.count, 0) : 0)}</div>
						{#if col.label === 'Lands'}
							{#each basics as c}
								<div class="mb-1 flex items-center gap-2 rounded bg-zinc-900 px-2 py-1 text-sm"><Pips colors={BASIC[c.name] ?? ''} /><span class="font-mono">{c.count}</span><span>{c.name}</span></div>
							{/each}
						{/if}
						<!-- ponytail: stacked images, each shows its title bar; hover for the full card -->
						<div class="relative" style="height: {col.cards.length ? `calc(${col.cards.length - 1} * 1.6rem + 11.2rem)` : '0'}">
							{#each col.cards as c, i}
								<img src={c.image?.[0]} alt={c.name} title="{c.name} · {pct(c.gihwr)}" loading="lazy"
									class="absolute w-32 rounded-lg shadow hover:z-10 hover:ring-2 hover:ring-emerald-400" style="top: {i * 1.6}rem"
									onmouseenter={(e) => show(c, e)} onmouseleave={hide} />
							{/each}
						</div>
					</div>
				{/each}
			</div>
			{#if b.note}<p class="mt-2 text-xs text-zinc-500">{b.note}</p>{/if}
		</section>
		<aside class="flex flex-col gap-4">
			<section>
				<h3 class="mb-1 text-sm font-semibold text-zinc-400">Sideboard ({count(b.sideboard)})</h3>
				{#each b.sideboard as c}
					<button class="flex w-full items-center gap-2 rounded px-1 text-left text-sm text-zinc-300 hover:bg-zinc-800" onmouseenter={(e) => show(c, e)} onmouseleave={hide}>
						<span class="flex w-10 shrink-0"><Pips colors={c.colors ?? []} /></span>
						<span class="w-4 font-mono text-xs text-zinc-500">{c.count}</span>
						<span class="min-w-0 flex-1 truncate">{c.name}</span>
						<span class="font-mono text-xs text-zinc-500">{pct(c.gihwr)}</span>
					</button>
				{/each}
			</section>
			<section>
				<h3 class="mb-1 text-sm font-semibold text-zinc-400">Simulator</h3>
				<table class="w-full font-mono text-sm"><tbody>
					{#each Object.entries(b.stats ?? {}) as [k, v]}
						<tr><td class="text-zinc-500">{k}</td><td class="text-right">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v}</td></tr>
					{/each}
				</tbody></table>
				<button class="mt-3 rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={copy}>{copied ? 'copied' : 'Copy Arena export'}</button>
			</section>
		</aside>
	</div>
{:else if ran && !busy}
	<p class="text-sm text-zinc-500">Upstream's builder returned no deck. It needs a near-complete pool (about 23 playable spells).</p>
{:else if !busy}
	<p class="text-sm text-zinc-500">Builds from the current pool with our deck builder (best 23 by predicted win rate, top players' deck shape). Needs a near-complete pool; after Arena restarts it uses the last draft's pool from Arena's event list.</p>
{/if}
{/if}
