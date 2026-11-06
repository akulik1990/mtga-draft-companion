<script lang="ts">
	import { page } from '$app/state';
	import Videos from '../../../lib/views/Videos.svelte';
	import VideoDetail from '../../../lib/views/VideoDetail.svelte';

	const id = $derived(page.params.id);
</script>

<svelte:head><title>Video drafts · Draft Companion</title></svelte:head>

<header class="sticky top-0 z-10 flex items-center gap-4 border-b border-zinc-800 bg-zinc-950/95 px-4 py-2 text-sm">
	<a href="/" class="text-zinc-400 hover:text-zinc-100">Draft Companion</a>
	<a href="/videos" class="rounded bg-zinc-800 px-3 py-1 text-zinc-100">Videos</a>
</header>

<main class="p-4">
	{#if id}
		{#key id}<VideoDetail {id} />{/key}
	{:else}
		<Videos />
	{/if}
</main>
