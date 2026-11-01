<script lang="ts">
	let { colors, size = 'h-3.5 w-3.5' }: { colors: string[] | string; size?: string } = $props();
	const FILL: Record<string, string> = { W: 'bg-amber-100', U: 'bg-sky-500', B: 'bg-zinc-600 ring-1 ring-zinc-400', R: 'bg-red-600', G: 'bg-green-600' };
	const list = $derived([...(typeof colors === 'string' ? colors : colors ?? [])].filter((c) => FILL[c]));
</script>

<span class="inline-flex gap-0.5 align-middle" title={list.join('')}>
	{#each list as c}<span class="inline-block rounded-full {size} {FILL[c]}"></span>{/each}
</span>
