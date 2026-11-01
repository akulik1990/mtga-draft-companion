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

<!-- ponytail: numbers sit over the art box (about 11-55% of a card's height); the name and rules text stay visible -->
<div class="relative flex flex-col overflow-hidden rounded-xl border bg-zinc-900 [container-type:inline-size] {highlight ? 'border-emerald-400 ring-2 ring-emerald-400/50' : 'border-zinc-800'}">
	<button type="button" {onclick} class="relative block cursor-pointer" title="{card.name}: details">
		<img src={card.image?.[0]} alt={card.name} loading="lazy" class="aspect-[5/7] w-full object-cover" />
		<div class="absolute inset-x-[6%] top-[11.5%] flex h-[43%] flex-col justify-between overflow-hidden rounded bg-zinc-950/80 p-[3%] text-left font-mono leading-tight" style="font-size: clamp(8px, 6.8cqw, 15px)"><!-- ponytail: text scales with card width -->
			<div class="flex flex-wrap items-center gap-1">
				{#if v2}<span class="rounded bg-violet-700 px-[0.3em] text-[1.15em] font-bold text-white" title="v2: rank {v2.rank}, score {v2.contextual_score.toFixed(2)}">V{v2.rank} <span class="font-normal text-violet-100">{v2.contextual_score.toFixed(1)}</span></span>{/if}
				{#if upstream}<span class="rounded bg-sky-900 px-1 text-sky-100" title="upstream: rank {upstream.rank}, score {upstream.contextual_score.toFixed(1)}">U{upstream.rank} {upstream.contextual_score.toFixed(0)}</span>{/if}
			</div>
			<div class="flex flex-wrap gap-1">
				{#if powerRank}<span class="rounded bg-amber-800 px-1 text-amber-100" title="Power: best card if you end up in its colours">Pow {powerRank}</span>{/if}
				{#if fitRank}<span class="rounded bg-emerald-800 px-1 text-emerald-100" title="Fit: best card for the colours your pool points to">Fit {fitRank}</span>{/if}
			</div>
			<div class="flex flex-wrap items-baseline gap-x-2">
				<span><span class="text-zinc-400">All</span> <span class="font-semibold {tone(snap, card, 'gihwr', ALL)}">{fmt(snap, card, 'gihwr', ALL)}</span></span>
				{#if archetype !== ALL}<span><span class="text-zinc-400">{archetype}</span> <span class="font-semibold {tone(snap, card, 'gihwr', archetype)}">{fmt(snap, card, 'gihwr', archetype)}</span></span>{/if}
				{#if wheel != null}<span class="text-zinc-300" title="wheel chance">↻{pct(wheel, 0)}</span>{/if}
				{#if icons}<span title={(card.tags ?? []).join(', ')}>{icons}</span>{/if}
			</div>
		</div>
	</button>
	{#if onpick}<button type="button" style="font-size: clamp(10px, 7cqw, 14px)" class="bg-emerald-700 py-[0.3em] font-semibold hover:bg-emerald-600 disabled:opacity-50" onclick={onpick} disabled={picking}>Pick</button>{/if}
</div>
