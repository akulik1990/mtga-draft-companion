<script lang="ts">
	import { onMount } from 'svelte';
	import { app, connect, refresh } from '../lib/state.svelte';
	import { api } from '../lib/api';
	import Pack from '../lib/views/Pack.svelte';
	import Analyst from '../lib/views/Analyst.svelte';
	import Pool from '../lib/views/Pool.svelte';
	import Deck from '../lib/views/Deck.svelte';
	import Settings from '../lib/views/Settings.svelte';
	import Compare from '../lib/views/Compare.svelte';
	import type { ResultFormat } from '../lib/api';
	import HoverImage from '../lib/HoverImage.svelte';

	const VIEWS = ['Draft', 'Analyst', 'Pool', 'Deck', 'Compare', 'Settings'] as const;
	const FORMATS: ResultFormat[] = ['Percentage', 'Rating', 'Grade'];
	const setting = (changes: Record<string, string>) => mock(() => api.updateSettings(changes));
	type View = (typeof VIEWS)[number];
	let view = $state<View>('Draft');

	onMount(() => { refresh(); connect(); });
	const snap = $derived(app.snap);
	let busy = $state(false);
	async function mock(fn: () => Promise<unknown>) {
		busy = true;
		try { await fn(); app.error = ''; } catch (e) { app.error = String(e); } finally { busy = false; refresh(); }
	}
</script>

<svelte:head><title>Draft Companion{snap?.pack_number ? ` · P${snap.pack_number}P${snap.pick_number}` : ''}</title></svelte:head>

<header class="sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-zinc-800 bg-zinc-950/95 px-4 py-2 text-sm">
	<span class="h-2 w-2 rounded-full {app.connected ? 'bg-emerald-400' : 'bg-rose-500'}" title={app.connected ? 'live' : 'disconnected'}></span>
	<span class="font-semibold whitespace-nowrap">{snap?.status ?? 'connecting…'}</span>
	{#if snap?.event?.set}<span class="text-zinc-400">{snap.event.set} {snap.event.type}</span>{/if}
	{#if snap?.dataset}<span class="hidden font-mono text-xs text-zinc-500 xl:inline">{snap.dataset}</span>{/if}
	{#if snap?.mock}
		<span class="rounded bg-violet-900/60 px-2 py-0.5 text-xs text-violet-200">mock {snap.mock.kind} {snap.mock.id}{snap.mock.done ? ' · finished' : ''}</span>
		<button class="text-xs text-zinc-400 hover:text-zinc-100" onclick={() => mock(api.mockStop)} disabled={busy}>end mock</button>
	{:else}
		<button class="rounded bg-violet-800 px-2 py-0.5 text-xs hover:bg-violet-700 disabled:opacity-50" onclick={() => mock(() => api.mockStart('random'))} disabled={busy} title="Draft against bots on the active set; a random seeded table or a replayed 17Lands draft">Mock draft</button>
	{/if}
	{#if snap?.deck_filter_options && Object.keys(snap.deck_filter_options).length}
		<label class="ml-auto flex items-center gap-1 text-xs text-zinc-500" title="Lane whose win rates the tables and grades show; Auto follows your pool, as upstream's Deck Filter does">
			deck filter
			<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" value={snap.deck_filter} onchange={(e) => setting({ deck_filter: e.currentTarget.value })} disabled={busy}>
				{#each Object.entries(snap.deck_filter_options) as [label, key] (key)}
					<option value={key}>{label}{key === 'Auto' && snap.deck_filter === 'Auto' ? ` → ${snap.archetype}` : ''}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-1 text-xs text-zinc-500" title="How win rates are shown, as upstream's Result Format setting">
			show
			<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" value={snap.result_format} onchange={(e) => setting({ result_format: e.currentTarget.value })} disabled={busy}>
				{#each FORMATS as f}<option value={f}>{f}</option>{/each}
			</select>
		</label>
	{/if}
	<nav class="{snap?.deck_filter_options ? '' : 'ml-auto'} flex gap-1">
		{#each VIEWS as v}
			<button class="rounded px-3 py-1 {view === v ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-100'}" onclick={() => (view = v)}>{v}</button>
		{/each}
		<a class="rounded px-3 py-1 text-zinc-400 hover:text-zinc-100" href="/videos/">Videos</a>
	</nav>
</header>

<main class="p-4">
	{#if app.error}<p class="mb-3 text-sm text-rose-400">{app.error}</p>{/if}
	{#if view === 'Settings'}
		<Settings />
	{:else if view === 'Deck'}
		<Deck />
	{:else if view === 'Compare'}
		<Compare {snap} />
	{:else if !snap}
		<p class="text-zinc-500">waiting for /api/state…</p>
	{:else if view === 'Analyst'}
		<Analyst {snap} />
	{:else if view === 'Pool'}
		<Pool {snap} />
	{:else}
		<Pack {snap} />
	{/if}
</main>
<HoverImage />
