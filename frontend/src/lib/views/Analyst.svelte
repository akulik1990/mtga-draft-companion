<script lang="ts">
	import type { Snapshot } from '../api';
	import { COLORS, COLOR_BG, isLand, pct } from '../cards';

	let { snap }: { snap: Snapshot } = $props();

	const sigMax = $derived(Math.max(1e-9, ...COLORS.map((c) => Math.abs(snap.signals[c] ?? 0))));
	const pairs = $derived(
		Object.entries(snap.color_ratings).filter(([k]) => k.length === 2).sort((a, b) => b[1] - a[1])
	);
	const spells = $derived(snap.pool.filter((c) => !isLand(c)));
	const curve = $derived.by(() => {
		const bins = [0, 0, 0, 0, 0, 0, 0, 0];
		for (const c of spells) bins[Math.min(7, c.cmc ?? 0)]++;
		return bins;
	});
	const curveMax = $derived(Math.max(1, ...curve));
	const interaction = $derived(spells.filter((c) => c.tags?.some((t) => /removal|counter|bounce|fight/.test(t))).length);
	const creatures = $derived(spells.filter((c) => c.types?.includes('Creature')).length);
</script>

<div class="grid gap-6 md:grid-cols-2">
	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Signals (packs 1 and 3, lateness × quality)</h3>
		{#each COLORS as c}
			{@const v = snap.signals[c] ?? 0}
			<div class="mb-1 flex items-center gap-2 text-sm">
				<span class="w-6 rounded text-center text-xs font-bold {COLOR_BG[c]}">{c}</span>
				<div class="h-3 flex-1 rounded bg-zinc-800"><div class="h-3 rounded bg-emerald-500" style="width:{(100 * Math.abs(v)) / sigMax}%"></div></div>
				<span class="w-12 text-right font-mono text-xs text-zinc-400">{v.toFixed(2)}</span>
			</div>
		{/each}
	</section>

	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Colour pair win rates (this dataset)</h3>
		<div class="grid grid-cols-2 gap-x-6 font-mono text-sm">
			{#each pairs as [k, v]}
				<div class="flex justify-between {k === snap.archetype ? 'text-emerald-300' : ''}"><span>{k}</span><span>{pct(v)}</span></div>
			{/each}
		</div>
	</section>

	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Pool curve ({spells.length} spells, {creatures} creatures, {interaction} interaction)</h3>
		<div class="flex items-end gap-1">
			{#each curve as n, i}
				<div class="flex flex-1 flex-col items-center gap-1">
					<span class="font-mono text-xs text-zinc-400">{n || ''}</span>
					<div class="w-full rounded-t bg-sky-600" style="height:{(72 * n) / curveMax}px"></div>
					<span class="font-mono text-xs text-zinc-500">{i === 7 ? '7+' : i}</span>
				</div>
			{/each}
		</div>
	</section>

	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">v2 lane distribution</h3>
		<p class="text-sm text-zinc-500">Phase 2. v2 currently mirrors upstream.</p>
	</section>

	{#each ['upstream', 'v2'] as key}
		<section>
			<h3 class="mb-2 text-sm font-semibold text-zinc-400">{key} top 3</h3>
			{#each (snap.advisors[key as 'upstream' | 'v2'] ?? []).slice(0, 3) as r}
				<div class="mb-2 text-sm">
					<span class="font-mono text-zinc-500">#{r.rank}</span> <span class="font-medium">{r.card_name}</span>
					<span class="font-mono text-zinc-400">{r.contextual_score.toFixed(1)} · base {pct(r.base_win_rate)} · z {r.z_score.toFixed(2)}</span>
					{#if r.reasoning?.length}<div class="text-xs text-zinc-500">{r.reasoning.join('; ')}</div>{/if}
				</div>
			{:else}
				<p class="text-sm text-zinc-600">no pack</p>
			{/each}
		</section>
	{/each}
</div>
