<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type VideoRow } from '../api';

	let rows = $state<VideoRow[]>([]);
	let error = $state('');
	onMount(async () => { try { rows = await api.videos(); } catch (e) { error = String(e); } });
	const pct = (a: number | null, n: number | null) => (a == null || !n ? '-' : `${a}/${n} (${Math.round((100 * a) / n)}%)`);
</script>

<h1 class="mb-3 text-lg font-semibold">Video drafts</h1>
{#if error}<p class="text-sm text-rose-400">{error}</p>{/if}
<table class="text-sm">
	<thead class="text-left text-xs text-zinc-500">
		<tr><th class="pr-6 pb-1 font-normal">Title</th><th class="pr-6 font-normal">Player</th><th class="pr-6 font-normal">Set</th><th class="pr-6 font-normal">v2 top-1</th><th class="font-normal">v2 top-3</th></tr>
	</thead>
	<tbody>
		{#each rows as v (v.id)}
			<tr class="border-t border-zinc-800">
				<td class="py-1 pr-6"><a class="text-sky-400 hover:underline" href="/videos/{v.id}">{v.title}</a></td>
				<td class="pr-6">{v.player}</td>
				<td class="pr-6">{v.set}</td>
				<td class="pr-6 tabular-nums">{pct(v.top1, v.n)}</td>
				<td class="tabular-nums">{pct(v.top3, v.n)}</td>
			</tr>
		{/each}
	</tbody>
</table>
