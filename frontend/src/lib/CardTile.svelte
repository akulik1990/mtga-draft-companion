<script lang="ts">
	import type { Card, Rec, Snapshot } from './api';
	import { ALL, pct } from './cards';
	import { fmt, tagIcons, tone } from './format';

	let {
		card, snap, archetype, upstream, v2, highlight = false, onclick, powerRank, fitRank
	}: { card: Card; snap: Snapshot; archetype: string; upstream?: Rec; v2?: Rec; highlight?: boolean; onclick: () => void; powerRank?: number; fitRank?: number } = $props();

	const wheel = $derived(upstream?.wheel_chance ?? v2?.wheel_chance);
	const icons = $derived(tagIcons(card.tags));
</script>

<button
	type="button"
	{onclick}
	class="group relative flex flex-col overflow-hidden rounded-lg border bg-zinc-900 text-left transition
		{highlight ? 'border-emerald-400 ring-2 ring-emerald-400/40' : 'border-zinc-800 hover:border-zinc-600'}"
>
	<img src={card.image?.[0]} alt={card.name} loading="lazy" class="aspect-[5/7] w-full object-cover" />
	<div class="absolute top-1 left-1 flex gap-1">
		{#if upstream}<span class="rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-sky-300" title="upstream: rank {upstream.rank}, score {upstream.contextual_score.toFixed(1)}">U{upstream.rank} · {upstream.contextual_score.toFixed(0)}</span>{/if}
		{#if v2}<span class="rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-violet-300" title="v2: rank {v2.rank}, score {v2.contextual_score.toFixed(2)}">V{v2.rank} · {v2.contextual_score.toFixed(1)}</span>{/if}
	</div>
	{#if wheel != null}
		<span class="absolute top-1 right-1 rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-zinc-400" title="wheel chance">↻{pct(wheel, 0)}</span>
	{/if}
	{#if powerRank || fitRank}
		<div class="absolute top-7 left-1 flex gap-1">
			{#if powerRank}<span class="rounded bg-amber-900/90 px-1.5 py-0.5 font-mono text-xs text-amber-200" title="Power: best card if you end up in its colours">Pow {powerRank}</span>{/if}
			{#if fitRank}<span class="rounded bg-emerald-900/90 px-1.5 py-0.5 font-mono text-xs text-emerald-200" title="Fit: best card for the colours your pool points to">Fit {fitRank}</span>{/if}
		</div>
	{/if}
	{#if icons}<span class="absolute bottom-10 right-1 rounded bg-zinc-950/90 px-1 text-xs" title={(card.tags ?? []).join(', ')}>{icons}</span>{/if}
	<div class="flex flex-col gap-0.5 px-2 py-1.5">
		<span class="truncate text-sm" title={card.name}>{card.name}</span>
		<span class="flex gap-2 font-mono text-xs">
			<span><span class="text-zinc-500">All</span> <span class="font-semibold {tone(snap, card, 'gihwr', ALL)}">{fmt(snap, card, 'gihwr', ALL)}</span></span>
			{#if archetype !== ALL}<span><span class="text-zinc-500">{archetype}</span> <span class="font-semibold {tone(snap, card, 'gihwr', archetype)}">{fmt(snap, card, 'gihwr', archetype)}</span></span>{/if}
		</span>
	</div>
</button>
