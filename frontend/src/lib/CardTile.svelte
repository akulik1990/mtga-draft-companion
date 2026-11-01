<script lang="ts">
	import type { Card, Rec } from './api';
	import { ALL, pct } from './cards';
	import Grade from './Grade.svelte';

	let {
		card, archetype, upstream, v2, highlight = false, onclick
	}: { card: Card; archetype: string; upstream?: Rec; v2?: Rec; highlight?: boolean; onclick: () => void } = $props();

	const g = $derived(card.grades ?? {});
	const wheel = $derived(upstream?.wheel_chance ?? v2?.wheel_chance);
</script>

<button
	type="button"
	{onclick}
	class="group relative flex flex-col overflow-hidden rounded-lg border bg-zinc-900 text-left transition
		{highlight ? 'border-emerald-400 ring-2 ring-emerald-400/40' : 'border-zinc-800 hover:border-zinc-600'}"
>
	<img src={card.image?.[0]} alt={card.name} loading="lazy" class="aspect-[5/7] w-full object-cover" />
	<div class="absolute top-1 left-1 flex gap-1">
		{#if upstream}<span class="rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-sky-300" title="upstream rank">U{upstream.rank}</span>{/if}
		{#if v2}<span class="rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-violet-300" title="v2 rank">V{v2.rank}</span>{/if}
	</div>
	{#if wheel != null}
		<span class="absolute top-1 right-1 rounded bg-zinc-950/90 px-1.5 py-0.5 font-mono text-xs text-zinc-400" title="wheel chance">↻{pct(wheel, 0)}</span>
	{/if}
	<div class="flex items-center justify-between gap-2 px-2 py-1.5">
		<span class="truncate text-sm">{card.name}</span>
		<span class="flex shrink-0 gap-2">
			<Grade grade={g[ALL]?.gihwr ?? '-'} label="All" />
			{#if archetype !== ALL}<Grade grade={g[archetype]?.gihwr ?? '-'} label={archetype} />{/if}
		</span>
	</div>
</button>
