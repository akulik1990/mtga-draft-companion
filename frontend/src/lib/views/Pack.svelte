<script lang="ts">
	import type { Card, Snapshot } from '../api';
	import { app, refresh } from '../state.svelte';
	import { api } from '../api';
	import { ALL, byName, gihwr, stats } from '../cards';
	import CardTile from '../CardTile.svelte';
	import CardDetail from '../CardDetail.svelte';

	let { snap }: { snap: Snapshot } = $props();
	type Sort = 'upstream' | 'v2' | 'gihwr' | 'ata' | 'alsa';
	let sort = $state<Sort>('upstream');
	let open = $state<Card | null>(null);
	let picking = $state(false);
	const mockLive = $derived(!!snap.mock && !snap.mock.done);
	async function pick(card: Card) {
		picking = true;
		try { await api.mockPick(card.name); open = null; app.error = ''; } catch (e) { app.error = String(e); } finally { picking = false; refresh(); }
	}

	const up = $derived(byName(snap.advisors.upstream));
	const v2 = $derived(byName(snap.advisors.v2));
	const top = $derived(snap.advisors[app.advisor]?.[0]?.card_name);
	const sorted = $derived.by(() => {
		const rank = (m: Map<string, { rank: number }>) => (c: Card) => m.get(c.name)?.rank ?? 99;
		const key: Record<Sort, (c: Card) => number> = {
			upstream: rank(up), v2: rank(v2),
			gihwr: (c) => -gihwr(c), ata: (c) => stats(c)?.ata ?? 99, alsa: (c) => stats(c)?.alsa ?? 99
		};
		return [...snap.pack].sort((a, b) => key[sort](a) - key[sort](b));
	});
</script>

{#if snap.mock?.done}
	<p class="p-8 text-center text-zinc-500">Mock draft finished: {snap.pool.length} cards. Build it in Deck, or end the mock.</p>
{:else if snap.pack.length === 0}
	<p class="p-8 text-center text-zinc-500">{snap.status}</p>
{:else}
	<div class="mb-3 flex items-center gap-3 text-sm">
		<span class="text-zinc-500">sort</span>
		{#each ['upstream', 'v2', 'gihwr', 'ata', 'alsa'] as s}
			<button class="rounded px-2 py-0.5 {sort === s ? 'bg-zinc-700' : 'text-zinc-400 hover:text-zinc-200'}" onclick={() => (sort = s as Sort)}>{s}</button>
		{/each}
		<span class="ml-auto text-zinc-500">archetype <span class="text-zinc-200">{snap.archetype}</span></span>
		{#if mockLive}
			<span class="text-violet-300">click a card, then Pick{snap.mock?.human ? ` · human took ${snap.mock.human}` : ''}</span>
		{/if}
		{#if snap.picked_from_pack.length}
			<span class="text-zinc-500">picked <span class="text-zinc-200">{snap.picked_from_pack.map((c) => c.name).join(', ')}</span></span>
		{/if}
	</div>
	<div class="grid grid-cols-3 gap-3 md:grid-cols-4 xl:grid-cols-5">
		{#each sorted as card (card.name)}
			<CardTile {card} archetype={snap.archetype} upstream={up.get(card.name)} v2={v2.get(card.name)}
				highlight={card.name === top} onclick={() => (open = card)} />
		{/each}
	</div>
{/if}

{#if open}
	<CardDetail card={open} archetype={snap.archetype} upstream={up.get(open.name)} v2={v2.get(open.name)} onclose={() => (open = null)}
		onpick={mockLive ? pick : undefined} {picking} />
{/if}
