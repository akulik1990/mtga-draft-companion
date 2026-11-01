<script lang="ts">
	import type { Card, Rec, Snapshot } from './api';
	import { ALL, pct } from './cards';
	import { fmt, tagIcons, tone } from './format';

	let {
		card, snap, archetype, upstream, v2, highlight = false, onclick, onpick, picking = false, powerRank, fitRank
	}: { card: Card; snap: Snapshot; archetype: string; upstream?: Rec; v2?: Rec; highlight?: boolean; onclick: () => void; onpick?: () => void; picking?: boolean; powerRank?: number; fitRank?: number } = $props();

	const wheel = $derived(upstream?.wheel_chance ?? v2?.wheel_chance);
	const icons = $derived(tagIcons(card.tags));
</script>

<div
	class="group relative flex flex-col overflow-hidden rounded-lg border bg-zinc-900 text-left transition
		{highlight ? 'border-emerald-400 ring-2 ring-emerald-400/40' : 'border-zinc-800 hover:border-zinc-600'}"
>
	<button type="button" {onclick} class="relative block cursor-pointer" title="details">
		<img src={card.image?.[0]} alt={card.name} loading="lazy" class="aspect-[5/7] w-full object-cover" />
		{#if icons}<span class="absolute right-1 bottom-1 rounded bg-zinc-950/90 px-1 text-sm" title={(card.tags ?? []).join(', ')}>{icons}</span>{/if}
	</button>
	<div class="flex flex-1 flex-col gap-1 px-2 py-1.5">
		<span class="truncate text-base font-medium" title={card.name}>{card.name}</span>
		<span class="flex flex-wrap gap-x-3 font-mono text-sm">
			<span><span class="text-zinc-500">All</span> <span class="font-semibold {tone(snap, card, 'gihwr', ALL)}">{fmt(snap, card, 'gihwr', ALL)}</span></span>
			{#if archetype !== ALL}<span><span class="text-zinc-500">{archetype}</span> <span class="font-semibold {tone(snap, card, 'gihwr', archetype)}">{fmt(snap, card, 'gihwr', archetype)}</span></span>{/if}
			{#if wheel != null}<span class="text-zinc-400" title="wheel chance">↻{pct(wheel, 0)}</span>{/if}
		</span>
		<span class="flex flex-wrap gap-1 font-mono text-sm">
			{#if v2}<span class="rounded bg-violet-950 px-1.5 text-violet-200" title="v2: rank {v2.rank}, score {v2.contextual_score.toFixed(2)}">V{v2.rank} · {v2.contextual_score.toFixed(1)}</span>{/if}
			{#if upstream}<span class="rounded bg-sky-950 px-1.5 text-sky-200" title="upstream: rank {upstream.rank}, score {upstream.contextual_score.toFixed(1)}">U{upstream.rank} · {upstream.contextual_score.toFixed(0)}</span>{/if}
			{#if powerRank}<span class="rounded bg-amber-950 px-1.5 text-amber-200" title="Power: best card if you end up in its colours">Pow {powerRank}</span>{/if}
			{#if fitRank}<span class="rounded bg-emerald-950 px-1.5 text-emerald-200" title="Fit: best card for the colours your pool points to">Fit {fitRank}</span>{/if}
		</span>
		{#if onpick}<button type="button" class="mt-auto rounded bg-emerald-700 py-1 text-sm font-semibold hover:bg-emerald-600 disabled:opacity-50" onclick={onpick} disabled={picking}>Pick</button>{/if}
	</div>
</div>
