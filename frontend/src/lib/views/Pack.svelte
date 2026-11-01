<script lang="ts">
	import type { Card, Snapshot } from '../api';
	import { app, refresh } from '../state.svelte';
	import { api } from '../api';
	import { byName, gihwr, stats } from '../cards';
	import { DEFAULT_COLUMNS, columns as allColumns, loadPref, savePref } from '../format';
	import CardTile from '../CardTile.svelte';
	import CardDetail from '../CardDetail.svelte';
	import CardTable from '../CardTable.svelte';
	import DraftPanel from '../DraftPanel.svelte';

	let { snap }: { snap: Snapshot } = $props();
	type Sort = 'upstream' | 'v2' | 'gihwr' | 'ata' | 'alsa';
	let sort = $state<Sort>('v2');
	let mode = $state<'cards' | 'table'>(loadPref('companion.packMode', 'cards'));
	let visible = $state<string[]>(loadPref('companion.columns', DEFAULT_COLUMNS));
	let picker = $state(false);
	let showSeen = $state(loadPref('companion.showSeen', true));
	let open = $state<Card | null>(null);
	let picking = $state(false);
	const mockLive = $derived(!!snap.mock && !snap.mock.done);
	async function pick(card: Card) {
		picking = true;
		try { await api.mockPick(card.name); open = null; app.error = ''; } catch (e) { app.error = String(e); } finally { picking = false; refresh(); }
	}

	const up = $derived(byName(snap.advisors.upstream));
	const v2 = $derived(byName(snap.advisors.v2));
	const cols = $derived(allColumns(snap, up, v2));
	const top = $derived(snap.advisors[app.advisor]?.[0]?.card_name);
	const sorted = $derived.by(() => {
		const rank = (m: Map<string, { rank: number }>) => (c: Card) => m.get(c.name)?.rank ?? 99;
		const key: Record<Sort, (c: Card) => number> = {
			upstream: rank(up), v2: rank(v2),
			gihwr: (c) => -gihwr(c), ata: (c) => stats(c)?.ata ?? 99, alsa: (c) => stats(c)?.alsa ?? 99
		};
		return [...snap.pack].sort((a, b) => key[sort](a) - key[sort](b));
	});
	const rankIn = (list: { card: string }[] | undefined, name: string) => { const i = (list ?? []).findIndex((x) => x.card === name); return i < 0 ? undefined : i + 1; };
	function setMode(m: 'cards' | 'table') { mode = m; savePref('companion.packMode', m); }
	function toggle(key: string) {
		visible = visible.includes(key) ? visible.filter((k) => k !== key) : [...visible, key];
		savePref('companion.columns', visible);
	}
	function toggleSeen() { showSeen = !showSeen; savePref('companion.showSeen', showSeen); }
</script>

{#if snap.mock?.done}
	<p class="p-8 text-center text-zinc-500">Mock draft finished: {snap.pool.length} cards. Build it in Deck, or end the mock.</p>
{:else if snap.pack.length === 0}
	<p class="p-8 text-center text-zinc-500">{snap.status}</p>
{:else}
<div class="grid gap-4 md:grid-cols-[minmax(0,1fr)_20rem] 2xl:grid-cols-[minmax(0,1fr)_24rem]">
<div>
	<div class="mb-3 flex flex-wrap items-center gap-3 text-sm">
		<span class="flex rounded bg-zinc-900">
			{#each ['cards', 'table'] as m}
				<button class="rounded px-2 py-0.5 {mode === m ? 'bg-zinc-700' : 'text-zinc-400 hover:text-zinc-200'}" onclick={() => setMode(m as 'cards' | 'table')}>{m}</button>
			{/each}
		</span>
		{#if mode === 'cards'}
			<span class="text-zinc-500">sort</span>
			{#each ['v2', 'upstream', 'gihwr', 'ata', 'alsa'] as s}
				<button class="rounded px-2 py-0.5 {sort === s ? 'bg-zinc-700' : 'text-zinc-400 hover:text-zinc-200'}" onclick={() => (sort = s as Sort)}>{s}</button>
			{/each}
		{:else}
			<button class="rounded px-2 py-0.5 text-zinc-400 hover:text-zinc-200" onclick={() => (picker = !picker)}>columns ▾</button>
		{/if}
		<span class="ml-auto text-zinc-500">values for <span class="text-zinc-200">{snap.archetype}</span></span>
		{#if mockLive}
			<span class="text-violet-300">Pick under a card, click it for details{snap.mock?.human ? ` · human took ${snap.mock.human}` : ''}</span>
		{/if}
		{#if snap.picked_from_pack.length}
			<span class="text-zinc-500">picked <span class="text-zinc-200">{snap.picked_from_pack.map((c) => c.name).join(', ')}</span></span>
		{/if}
	</div>
	{#if snap.v2_split?.branch}
		{@const b = snap.v2_split.branch}
		<div class="mb-3 rounded border px-3 py-2 text-sm {b.take ? 'border-amber-500/60 bg-amber-950/40' : 'border-zinc-700 bg-zinc-900'}">
			<span class="text-zinc-400">Strongest card:</span> <span class="font-semibold">{b.card}</span>
			<span class="text-zinc-400"> · {snap.pool.length ? 'best for your colours' : 'most flexible (playable in the most colour pairs)'}:</span> <span class="font-semibold">{b.over}</span>
			<span class="text-zinc-400"> · gap</span> <span class="font-mono">{b.gap >= 0 ? '+' : ''}{b.gap.toFixed(1)}</span>
			<span class="text-zinc-400">; winning top drafters switch above</span> <span class="font-mono">{b.threshold >= 0 ? '+' : ''}{b.threshold.toFixed(1)}</span>
			<span class="text-zinc-500">(pool {b.stage})</span>
			→ <span class="font-semibold {b.take ? 'text-amber-300' : 'text-emerald-300'}">{b.take ? `take ${b.card}` : `stay with ${b.over}`}</span>
		</div>
	{/if}
	{#if mode === 'table' && picker}
		<div class="mb-3 flex flex-wrap gap-2 rounded bg-zinc-900 p-2 text-xs">
			{#each cols as c (c.key)}
				<label class="flex items-center gap-1" title={c.title}><input type="checkbox" checked={visible.includes(c.key)} onchange={() => toggle(c.key)} />{c.label}</label>
			{/each}
		</div>
	{/if}
	{#if mode === 'cards'}
		<div class="grid grid-cols-7 gap-2">
			{#each sorted as card (card.name)}
				<CardTile {card} {snap} archetype={snap.archetype} upstream={up.get(card.name)} v2={v2.get(card.name)}
					powerRank={rankIn(snap.v2_split?.power, card.name)} fitRank={rankIn(snap.v2_split?.fit, card.name)}
					highlight={card.name === top} onclick={() => (open = card)}
					onpick={mockLive ? () => pick(card) : undefined} {picking} />
			{/each}
		</div>
	{:else}
		<CardTable cards={sorted} columns={cols} {visible} highlight={top} onrow={(c) => (open = c)} />
	{/if}

	{#if snap.missing.length}
		<section class="mt-5">
			<button class="mb-1 text-sm font-semibold text-zinc-400 hover:text-zinc-200" onclick={toggleSeen}>
				{showSeen ? '▾' : '▸'} Seen cards, taken since this pack was last here ({snap.missing.length})
			</button>
			{#if showSeen}
				<CardTable cards={snap.missing} columns={cols} visible={visible.filter((k) => k !== 'upstream' && k !== 'v2' && k !== 'wheel')} onrow={(c) => (open = c)} />
			{/if}
		</section>
	{/if}
</div>
<div class="md:sticky md:top-24 md:max-h-[calc(100vh-6.5rem)] md:overflow-y-auto"><DraftPanel {snap} onopen={(c) => (open = c)} /></div>
</div>
{/if}

{#if open}
	<CardDetail card={open} archetype={snap.archetype} upstream={up.get(open.name)} v2={v2.get(open.name)} onclose={() => (open = null)}
		onpick={mockLive && snap.pack.some((c) => c.name === open?.name) ? pick : undefined} {picking} />
{/if}
