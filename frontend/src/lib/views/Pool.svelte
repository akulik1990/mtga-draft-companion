<script lang="ts">
	import type { Card, Snapshot } from '../api';
	import { ALL, COLOR_BG, colorGroup } from '../cards';
	import Grade from '../Grade.svelte';
	import CardDetail from '../CardDetail.svelte';

	let { snap }: { snap: Snapshot } = $props();
	let open = $state<Card | null>(null);

	const ORDER = ['W', 'U', 'B', 'R', 'G', 'Multi', 'Colorless', 'Land'];
	const groups = $derived.by(() => {
		const m = new Map<string, Card[]>();
		for (const c of snap.pool) m.set(colorGroup(c), [...(m.get(colorGroup(c)) ?? []), c]);
		return ORDER.filter((k) => m.has(k)).map((k) => [k, m.get(k)!.sort((a, b) => a.cmc - b.cmc || a.name.localeCompare(b.name))] as const);
	});
</script>

{#snippet row(c: Card)}
	<button class="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left text-sm hover:bg-zinc-800" onclick={() => (open = c)}>
		<span class="w-4 font-mono text-xs text-zinc-500">{c.cmc}</span>
		<span class="flex-1 truncate">{c.name}</span>
		<span class="truncate text-xs text-zinc-600">{c.types?.join(' ')}</span>
		<Grade grade={c.grades?.[ALL]?.gihwr ?? '-'} />
		{#if snap.archetype !== ALL}<Grade grade={c.grades?.[snap.archetype]?.gihwr ?? '-'} label={snap.archetype} />{/if}
	</button>
{/snippet}

<div class="grid gap-6 md:grid-cols-[2fr_1fr]">
	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Pool ({snap.pool.length})</h3>
		{#each groups as [k, cards]}
			<div class="mb-3">
				<div class="mb-1 flex items-center gap-2 text-xs"><span class="rounded px-1.5 font-bold {COLOR_BG[k] ?? 'bg-zinc-700'}">{k}</span><span class="text-zinc-500">{cards.length}</span></div>
				{#each cards as c}{@render row(c)}{/each}
			</div>
		{:else}
			<p class="text-sm text-zinc-600">nothing taken yet</p>
		{/each}
	</section>
	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Missing from this pack ({snap.missing.length})</h3>
		{#each snap.missing as c}{@render row(c)}{:else}<p class="text-sm text-zinc-600">first look at this pack</p>{/each}
	</section>
</div>

{#if open}<CardDetail card={open} archetype={snap.archetype} onclose={() => (open = null)} />{/if}
