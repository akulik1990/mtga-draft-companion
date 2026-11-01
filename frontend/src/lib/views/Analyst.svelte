<script lang="ts">
	import type { Snapshot } from '../api';
	import { COLORS, COLOR_BG, isLand, pct } from '../cards';

	let { snap }: { snap: Snapshot } = $props();

	const SIDE_LABEL: Record<string, string> = { right: 'packs 1 and 3 neighbour', left: 'pack 2 neighbour' };
	const sides = $derived((['right', 'left'] as const).filter((k) => snap.signals_v2_sides?.[k]));
	const sideMax = $derived(Math.max(1e-9, ...sides.flatMap((k) => COLORS.map((c) => Math.abs(snap.signals_v2_sides?.[k]?.[c] ?? 0)))));
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
	// Upstream's type breakdown (dashboard.update_deck_balance), basics excluded.
	const TYPES = ['Creature', 'Planeswalker', 'Battle', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Land'];
	const typeCounts = $derived.by(() => {
		const out: Record<string, number> = Object.fromEntries(TYPES.map((t) => [t, 0]));
		for (const c of snap.pool) {
			const types = c.types ?? [];
			if (types.includes('Basic')) continue;
			const t = TYPES.find((x) => types.includes(x));
			if (t) out[t] += 1;
		}
		return out;
	});
	const typeMax = $derived(Math.max(1, ...Object.values(typeCounts)));
	const lanes = $derived(Object.entries(snap.lanes ?? {}).sort((a, b) => b[1] - a[1]));
	const laneMax = $derived(Math.max(1e-9, ...lanes.map(([, p]) => p)));
</script>

<div class="grid gap-6 md:grid-cols-2">
	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Upstream signals (packs 1 and 3, lateness × quality)</h3>
		{#each COLORS as c}
			{@const v = snap.signals[c] ?? 0}
			<div class="mb-1 flex items-center gap-2 text-sm">
				<span class="w-6 rounded text-center text-xs font-bold {COLOR_BG[c]}">{c}</span>
				<div class="h-3 flex-1 rounded bg-zinc-800"><div class="h-3 rounded bg-emerald-500" style="width:{(100 * Math.abs(v)) / sigMax}%"></div></div>
				<span class="w-12 text-right font-mono text-xs text-zinc-400">{v.toFixed(2)}</span>
			</div>
		{/each}
	</section>

	{#if sides.length}
		<section>
			<h3 class="mb-2 text-sm font-semibold text-zinc-400">v2 signals by neighbour (in use: {SIDE_LABEL[snap.signals_v2_sides?.active ?? 'right']})</h3>
			{#each sides as side}
				<div class="mb-1 text-xs {snap.signals_v2_sides?.active === side ? 'text-zinc-200' : 'text-zinc-500'}">{SIDE_LABEL[side]}</div>
				{#each COLORS as c}
					{@const v = snap.signals_v2_sides?.[side]?.[c] ?? 0}
					<div class="mb-0.5 flex items-center gap-2 text-sm">
						<span class="w-6 rounded text-center text-xs font-bold {COLOR_BG[c]}">{c}</span>
						<div class="relative h-2 flex-1 rounded bg-zinc-800">
							<div class="absolute h-2 rounded {v >= 0 ? 'bg-violet-500' : 'bg-zinc-600'}" style="left:{v >= 0 ? 50 : 50 - (50 * Math.abs(v)) / sideMax}%; width:{(50 * Math.abs(v)) / sideMax}%"></div>
						</div>
						<span class="w-12 text-right font-mono text-xs text-zinc-400">{v.toFixed(2)}</span>
					</div>
				{/each}
			{/each}
		</section>
	{/if}

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
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Pool by type</h3>
		{#each TYPES.filter((t) => typeCounts[t] > 0) as t}
			<div class="mb-1 flex items-center gap-2 text-sm">
				<span class="w-24 text-xs text-zinc-400">{t}</span>
				<div class="h-3 flex-1 rounded bg-zinc-800"><div class="h-3 rounded bg-amber-500" style="width:{(100 * typeCounts[t]) / typeMax}%"></div></div>
				<span class="w-8 text-right font-mono text-xs text-zinc-400">{typeCounts[t]}</span>
			</div>
		{:else}
			<p class="text-sm text-zinc-600">nothing taken yet</p>
		{/each}
	</section>

	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">v2 P(lane) from pool, signals and format</h3>
		{#each lanes.slice(0, 6) as [lane, p]}
			<div class="mb-1 flex items-center gap-2 text-sm">
				<span class="w-8 font-mono text-xs {lane === snap.archetype ? 'text-emerald-300' : 'text-zinc-400'}">{lane}</span>
				<div class="h-3 flex-1 rounded bg-zinc-800"><div class="h-3 rounded bg-violet-500" style="width:{(100 * p) / laneMax}%"></div></div>
				<span class="w-12 text-right font-mono text-xs text-zinc-400">{(100 * p).toFixed(0)}%</span>
			</div>
		{:else}
			<p class="text-sm text-zinc-600">no pack</p>
		{/each}
	</section>

	{#if snap.v2_split}
		<section>
			<h3 class="mb-2 text-sm font-semibold text-zinc-400">v2 Power top 3 (best if you end up in its colours)</h3>
			{#each snap.v2_split.power as x, i}<div class="text-sm"><span class="font-mono text-zinc-500">#{i + 1}</span> {x.card} <span class="font-mono text-zinc-400">{x.score.toFixed(1)}</span></div>{/each}
		</section>
		<section>
			<h3 class="mb-2 text-sm font-semibold text-zinc-400">v2 Fit top 3 ({snap.pool.length ? 'best for your colours' : 'most flexible: no colours yet'})</h3>
			{#each snap.v2_split.fit as x, i}<div class="text-sm"><span class="font-mono text-zinc-500">#{i + 1}</span> {x.card} <span class="font-mono text-zinc-400">{x.score.toFixed(1)}</span></div>{/each}
		</section>
	{/if}

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
