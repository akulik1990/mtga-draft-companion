<script lang="ts">
	import { onMount } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { api, type VideoRow } from '../api';

	type Key = 'title' | 'player' | 'set' | 'top1' | 'top3' | 'added';
	const COLS: [Key, string][] = [['title', 'Title'], ['player', 'Player'], ['set', 'Set'], ['top1', 'v2 top-1'], ['top3', 'v2 top-3'], ['added', 'Added']];

	let rows = $state<VideoRow[]>([]);
	let error = $state('');
	let set = $state('');
	let player = $state('');
	let q = $state('');
	let sort = $state<Key>('added');
	let desc = $state(true);
	let ready = $state(false);

	onMount(async () => {
		try {
			const p = new URLSearchParams(location.search);
			set = p.get('set') ?? ''; player = p.get('player') ?? ''; q = p.get('q') ?? '';
			const s = p.get('sort') as Key | null;
			if (s && COLS.some(([k]) => k === s)) { sort = s; desc = p.get('dir') !== 'asc'; }
		} catch { /* defaults */ }
		ready = true;
		try { rows = await api.videos(); } catch (e) { error = String(e); }
	});

	// keep the choices in the query; `v` is the detail route and is never present here
	$effect(() => {
		const p = new URLSearchParams();
		if (set) p.set('set', set);
		if (player) p.set('player', player);
		if (q) p.set('q', q);
		if (sort !== 'added' || !desc) { p.set('sort', sort); p.set('dir', desc ? 'desc' : 'asc'); }
		if (!ready) return;
		try { replaceState(p.size ? `?${p}` : location.pathname, {}); } catch { /* router not ready */ }
	});

	const uniq = (f: (v: VideoRow) => string) => [...new Set(rows.map(f).filter(Boolean))].sort();
	const sets = $derived(uniq((v) => v.set));
	const players = $derived(uniq((v) => v.player));
	const val = (v: VideoRow, k: Key): string | number => (k === 'top1' || k === 'top3' ? (v[k] == null || !v.n ? -1 : v[k]! / v.n) : (v[k] ?? ''));
	const shown = $derived.by(() => {
		const t = q.trim().toLowerCase();
		const out = rows.filter((v) => (!set || v.set === set) && (!player || v.player === player) && (!t || v.title.toLowerCase().includes(t)));
		return out.sort((a, b) => {
			const x = val(a, sort), y = val(b, sort);
			return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1);
		});
	});
	const click = (k: Key) => { if (sort === k) desc = !desc; else { sort = k; desc = k !== 'title' && k !== 'player' && k !== 'set'; } };
	const clear = () => { set = ''; player = ''; q = ''; };
	const pct = (a: number | null, n: number | null) => (a == null || !n ? '-' : `${a}/${n} (${Math.round((100 * a) / n)}%)`);
	const sel = 'rounded bg-zinc-900 px-1 py-0.5 text-zinc-200';
</script>

<h1 class="mb-3 text-lg font-semibold">Video drafts</h1>
{#if error}<p class="text-sm text-rose-400">{error}</p>{/if}
<div class="mb-3 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
	<select class={sel} bind:value={set} aria-label="Set"><option value="">All sets</option>{#each sets as s}<option>{s}</option>{/each}</select>
	<select class={sel} bind:value={player} aria-label="Player"><option value="">All players</option>{#each players as s}<option>{s}</option>{/each}</select>
	<input class="{sel} min-w-0 flex-1 basis-40" type="search" placeholder="search title" bind:value={q} aria-label="Search title" />
	{#if set || player || q}<button class="text-sky-400 hover:underline" onclick={clear}>clear</button>{/if}
	<span class="tabular-nums">{shown.length} of {rows.length}</span>
</div>
<div class="overflow-x-auto">
	<table class="text-sm">
		<thead class="text-left text-xs text-zinc-500">
			<tr>
				{#each COLS as [k, label]}
					<th class="pr-6 pb-1 font-normal whitespace-nowrap" aria-sort={sort === k ? (desc ? 'descending' : 'ascending') : 'none'}>
						<button class="hover:text-zinc-100 {sort === k ? 'text-zinc-100' : ''}" onclick={() => click(k)}>{label}{sort === k ? (desc ? ' ▼' : ' ▲') : ''}</button>
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each shown as v (v.id)}
				<tr class="border-t border-zinc-800">
					<td class="py-1 pr-6"><a class="text-sky-400 hover:underline" href="?v={v.id}">{v.title}</a></td>
					<td class="pr-6">{v.player}</td>
					<td class="pr-6">{v.set}</td>
					<td class="pr-6 whitespace-nowrap tabular-nums">{pct(v.top1, v.n)}</td>
					<td class="pr-6 whitespace-nowrap tabular-nums">{pct(v.top3, v.n)}</td>
					<td class="whitespace-nowrap tabular-nums">{v.added?.slice(0, 10) ?? ''}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
