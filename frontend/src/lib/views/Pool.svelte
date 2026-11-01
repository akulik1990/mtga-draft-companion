<script lang="ts">
	import type { Card, Snapshot } from '../api';
	import { ALL, COLOR_BG, colorGroup } from '../cards';
	import { fmt, loadPref, savePref, tagIcons, tone } from '../format';
	import CardDetail from '../CardDetail.svelte';
	import Pips from '../Pips.svelte';
	import { show, hide } from '../hover.svelte';

	let { snap }: { snap: Snapshot } = $props();
	let open = $state<Card | null>(null);
	// Like upstream's Card Pool "Colors" dropdown: follow the deck filter, or look at the pool under any lane.
	let override = $state<string>(loadPref('companion.poolLane', ''));
	const lane = $derived(override && snap.metrics?.[override] ? override : snap.archetype || ALL);
	const lanes = $derived(Object.keys(snap.metrics ?? {}).filter((l) => l !== ALL).sort((a, b) => a.length - b.length || a.localeCompare(b)));
	function choose(v: string) { override = v; savePref('companion.poolLane', v); }

	const ORDER = ['W', 'U', 'B', 'R', 'G', 'Multi', 'Colorless', 'Land'];
	const groups = $derived.by(() => {
		const m = new Map<string, Card[]>();
		for (const c of snap.pool) m.set(colorGroup(c), [...(m.get(colorGroup(c)) ?? []), c]);
		return ORDER.filter((k) => m.has(k)).map((k) => [k, m.get(k)!.sort((a, b) => a.cmc - b.cmc || a.name.localeCompare(b.name))] as const);
	});
</script>

{#snippet row(c: Card)}
	<button class="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left text-sm hover:bg-zinc-800" onclick={() => (open = c)} onmouseenter={(e) => show(c, e)} onmouseleave={hide}>
		<span class="flex w-10 shrink-0"><Pips colors={c.colors ?? []} /></span>
		<span class="w-4 font-mono text-xs text-zinc-500">{c.cmc}</span>
		<span class="flex-1 truncate">{c.name}</span>
		<span class="text-xs">{tagIcons(c.tags)}</span>
		<span class="truncate text-xs text-zinc-600">{c.types?.join(' ')}</span>
		<span class="w-20 text-right font-mono text-xs"><span class="text-zinc-500">All</span> <span class={tone(snap, c, 'gihwr', ALL)}>{fmt(snap, c, 'gihwr', ALL)}</span></span>
		{#if lane !== ALL}<span class="w-20 text-right font-mono text-xs"><span class="text-zinc-500">{lane}</span> <span class={tone(snap, c, 'gihwr', lane)}>{fmt(snap, c, 'gihwr', lane)}</span></span>{/if}
	</button>
{/snippet}

<div class="grid gap-6 md:grid-cols-[2fr_1fr]">
	<section>
		<div class="mb-2 flex items-center gap-3">
			<h3 class="text-sm font-semibold text-zinc-400">Pool ({snap.pool.length})</h3>
			<label class="ml-auto flex items-center gap-1 text-xs text-zinc-500">
				colours
				<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" value={override} onchange={(e) => choose(e.currentTarget.value)}>
					<option value="">Follow deck filter ({snap.archetype})</option>
					{#each lanes as l}<option value={l}>{l}</option>{/each}
				</select>
			</label>
		</div>
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

{#if open}<CardDetail card={open} archetype={lane} onclose={() => (open = null)} />{/if}
