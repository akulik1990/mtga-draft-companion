<script lang="ts">
	import type { Card, Snapshot } from './api';
	import { ALL, COLORS, colorGroup, isLand } from './cards';
	import { fmt, tone } from './format';
	import Pips from './Pips.svelte';

	let { snap, onopen }: { snap: Snapshot; onopen: (c: Card) => void } = $props();

	const lanes = $derived(Object.entries(snap.lanes ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 4));
	const side = $derived(snap.signals_v2_sides?.[(snap.signals_v2_sides?.active ?? 'right') as 'right' | 'left'] ?? snap.signals_v2 ?? {});
	const spells = $derived(snap.pool.filter((c) => !isLand(c)));
	const curve = $derived.by(() => {
		const bins = [0, 0, 0, 0, 0, 0, 0];
		for (const c of spells) bins[Math.max(0, Math.min(6, (c.cmc ?? 0) - 1))]++;
		return bins;
	});
	const curveMax = $derived(Math.max(1, ...curve));
	const creatures = $derived(spells.filter((c) => c.types?.includes('Creature')).length);
	const interaction = $derived(spells.filter((c) => c.tags?.some((t) => /removal|counter|bounce|fight/.test(t))).length);
	const ORDER = ['W', 'U', 'B', 'R', 'G', 'Multi', 'Colorless', 'Land'];
	const groups = $derived.by(() => {
		const m = new Map<string, Card[]>();
		for (const c of snap.pool) m.set(colorGroup(c), [...(m.get(colorGroup(c)) ?? []), c]);
		return ORDER.filter((k) => m.has(k)).map((k) => [k, m.get(k)!.sort((a, b) => a.cmc - b.cmc || a.name.localeCompare(b.name))] as const);
	});
	const lane = $derived(snap.archetype || ALL);

	let hover = $state<{ card: Card; x: number; y: number } | null>(null);
	let panel: HTMLElement;
	// The image opens left of the panel, level with the row, so it never covers the list.
	const show = (card: Card, e: MouseEvent) => (hover = { card, x: panel.getBoundingClientRect().left, y: (e.currentTarget as HTMLElement).getBoundingClientRect().top });
</script>

<aside bind:this={panel} class="flex flex-col gap-4 text-sm">
	<section>
		<h3 class="mb-1 font-semibold text-zinc-400">Lanes (v2)</h3>
		{#each lanes as [l, p]}
			<div class="mb-1 flex items-center gap-2">
				<Pips colors={l} />
				<div class="h-2.5 flex-1 rounded bg-zinc-800"><div class="h-2.5 rounded bg-violet-500" style="width:{100 * p}%"></div></div>
				<span class="w-10 text-right font-mono text-zinc-300">{(100 * p).toFixed(0)}%</span>
			</div>
		{:else}<p class="text-zinc-600">no pack</p>{/each}
	</section>

	<section>
		<h3 class="mb-1 font-semibold text-zinc-400" title="v2's signal from the neighbour passing to you now: positive = flowing">Open colours</h3>
		<div class="flex justify-between">
			{#each COLORS as c}
				{@const v = side[c] ?? 0}
				<span class="flex flex-col items-center gap-0.5"><Pips colors={c} size="h-4 w-4" /><span class="font-mono {v > 0.3 ? 'text-emerald-400' : v < -0.3 ? 'text-rose-400' : 'text-zinc-400'}">{v >= 0 ? '+' : ''}{v.toFixed(1)}</span></span>
			{/each}
		</div>
	</section>

	<section>
		<h3 class="mb-1 font-semibold text-zinc-400">Pool {snap.pool.length} <span class="font-normal text-zinc-500">· {creatures} creatures · {interaction} interaction</span></h3>
		<div class="mt-1 mb-3 flex h-16 items-end gap-1">
			{#each curve as n, i}
				<div class="flex flex-1 flex-col items-center">
					<span class="font-mono text-xs text-zinc-400">{n || ''}</span>
					<div class="w-full rounded-t bg-sky-600" style="height:{(32 * n) / curveMax}px"></div>
					<span class="font-mono text-xs text-zinc-500">{i === 6 ? '7+' : i + 1}</span>
				</div>
			{/each}
		</div>
		{#each groups as [, cards]}
			<div class="mb-2 border-t border-zinc-800 pt-1">
				{#each cards as c}
					<button class="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-zinc-800" onclick={() => onopen(c)}
						onmouseenter={(e) => show(c, e)} onmouseleave={() => (hover = null)}>
						<span class="flex w-10 shrink-0"><Pips colors={c.colors ?? []} /></span>
						<span class="w-4 shrink-0 text-right font-mono text-xs text-zinc-500">{c.cmc}</span>
						<span class="min-w-0 flex-1 truncate" title={c.name}>{c.name}</span>
						<span class="w-12 shrink-0 text-right font-mono text-xs {tone(snap, c, 'gihwr', lane)}">{fmt(snap, c, 'gihwr', lane)}</span>
					</button>
				{/each}
			</div>
		{:else}<p class="text-zinc-600">nothing taken yet</p>{/each}
	</section>
</aside>

{#if hover?.card.image?.[0]}
	<img src={hover.card.image[0]} alt="" class="pointer-events-none fixed z-50 w-64 rounded-xl shadow-2xl"
		style="left:{Math.max(8, hover.x - 272)}px; top:{Math.max(8, Math.min(hover.y - 150, innerHeight - 370))}px" />
{/if}
