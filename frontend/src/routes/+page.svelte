<script lang="ts">
	import { onMount } from 'svelte';
	import { app, connect, refresh } from '../lib/state.svelte';
	import Pack from '../lib/views/Pack.svelte';
	import Analyst from '../lib/views/Analyst.svelte';
	import Pool from '../lib/views/Pool.svelte';
	import Deck from '../lib/views/Deck.svelte';
	import Settings from '../lib/views/Settings.svelte';

	const VIEWS = ['Pack', 'Analyst', 'Pool', 'Deck', 'Settings'] as const;
	type View = (typeof VIEWS)[number];
	let view = $state<View>('Pack');

	onMount(() => { refresh(); connect(); });
	const snap = $derived(app.snap);
</script>

<svelte:head><title>Draft Companion{snap?.pack_number ? ` · P${snap.pack_number}P${snap.pick_number}` : ''}</title></svelte:head>

<header class="sticky top-0 z-10 flex items-center gap-4 border-b border-zinc-800 bg-zinc-950/95 px-4 py-2 text-sm">
	<span class="h-2 w-2 rounded-full {app.connected ? 'bg-emerald-400' : 'bg-rose-500'}" title={app.connected ? 'live' : 'disconnected'}></span>
	<span class="font-semibold">{snap?.status ?? 'connecting…'}</span>
	{#if snap?.event?.set}<span class="text-zinc-400">{snap.event.set} {snap.event.type}</span>{/if}
	{#if snap?.dataset}<span class="font-mono text-xs text-zinc-500">{snap.dataset}</span>{/if}
	<nav class="ml-auto flex gap-1">
		{#each VIEWS as v}
			<button class="rounded px-3 py-1 {view === v ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-100'}" onclick={() => (view = v)}>{v}</button>
		{/each}
	</nav>
</header>

<main class="p-4">
	{#if app.error}<p class="mb-3 text-sm text-rose-400">{app.error}</p>{/if}
	{#if view === 'Settings'}
		<Settings />
	{:else if view === 'Deck'}
		<Deck />
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
