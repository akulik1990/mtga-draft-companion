<script lang="ts">
	import { api, type Card, type Snapshot } from '../api';
	import { ALL } from '../cards';
	import { FIELD_LABEL, FIELD_TITLE, WR_FIELDS, fmt, loadPref, savePref, stripe, tagIcons, tone } from '../format';
	import CardDetail from '../CardDetail.svelte';

	// Upstream's Compare tab: search the set, line cards up, read every number side by side.
	let { snap }: { snap: Snapshot | null } = $props();
	let q = $state('');
	let found = $state<Card[]>([]);
	let list = $state<Card[]>(loadPref('companion.compare', []));
	let open = $state<Card | null>(null);
	let lane = $state('');
	const active = $derived(lane || snap?.archetype || ALL);
	const lanes = $derived(Object.keys(snap?.metrics ?? {}).sort((a, b) => (a === ALL ? -1 : b === ALL ? 1 : a.length - b.length || a.localeCompare(b))));
	const FIELDS = [...WR_FIELDS, 'iwd', 'alsa', 'ata'];

	let timer: ReturnType<typeof setTimeout>;
	function search(v: string) {
		q = v;
		clearTimeout(timer);
		timer = setTimeout(async () => { found = q.trim().length >= 2 ? await api.cards(q) : []; }, 200);
	}
	function add(c: Card) {
		if (!list.some((x) => x.name === c.name)) list = [...list, c];
		savePref('companion.compare', list);
		q = ''; found = [];
	}
	function remove(name: string) { list = list.filter((c) => c.name !== name); savePref('companion.compare', list); }
	function clear() { list = []; savePref('companion.compare', list); }
</script>

{#if !snap}
	<p class="text-zinc-500">waiting for /api/state…</p>
{:else}
	<div class="mb-3 flex flex-wrap items-center gap-3 text-sm">
		<div class="relative w-80">
			<input class="w-full rounded bg-zinc-900 px-2 py-1" placeholder="search the set's cards" value={q} oninput={(e) => search(e.currentTarget.value)} />
			{#if found.length}
				<div class="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded bg-zinc-900 shadow-lg">
					{#each found as c (c.name)}
						<button class="block w-full px-2 py-1 text-left hover:bg-zinc-800" onclick={() => add(c)}>{c.name} <span class="text-xs text-zinc-500">{(c.colors ?? []).join('')}</span></button>
					{/each}
				</div>
			{/if}
		</div>
		<label class="flex items-center gap-1 text-xs text-zinc-500">
			lane
			<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" value={lane} onchange={(e) => (lane = e.currentTarget.value)}>
				<option value="">Follow deck filter ({snap.archetype})</option>
				{#each lanes as l}<option value={l}>{l}</option>{/each}
			</select>
		</label>
		{#if list.length}<button class="text-xs text-zinc-400 hover:text-zinc-100" onclick={clear}>clear</button>{/if}
	</div>

	{#if list.length}
		<table class="w-full font-mono text-xs">
			<thead class="text-zinc-500">
				<tr><th class="px-2 py-1 text-left font-normal">card</th>
					{#each FIELDS as f}<th class="px-2 py-1 text-right font-normal" title={FIELD_TITLE[f]}>{FIELD_LABEL[f]}</th>{/each}
					<th class="px-2 py-1 text-right font-normal">TAGS</th><th></th></tr>
			</thead>
			<tbody>
				{#each list as c (c.name)}
					<tr class="border-l-4 {stripe(c)} hover:bg-zinc-800/60">
						<td class="px-2 py-1 font-sans text-sm"><button class="hover:underline" onclick={() => (open = c)}>{c.name}</button></td>
						{#each FIELDS as f}<td class="px-2 py-1 text-right {tone(snap, c, f, active)}">{fmt(snap, c, f, active)}</td>{/each}
						<td class="px-2 py-1 text-right">{tagIcons(c.tags) || '-'}</td>
						<td class="px-2 py-1 text-right"><button class="text-zinc-500 hover:text-rose-400" onclick={() => remove(c.name)}>✕</button></td>
					</tr>
				{/each}
			</tbody>
		</table>
	{:else}
		<p class="text-sm text-zinc-500">Search for cards of the active set to line them up side by side under any lane.</p>
	{/if}
{/if}

{#if open}<CardDetail card={open} archetype={active} onclose={() => (open = null)} />{/if}
