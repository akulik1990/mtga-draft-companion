<script lang="ts">
	import { gradeTone } from './cards';
	let { grade, label = '' }: { grade: string; label?: string } = $props();
</script>

<span class="inline-flex items-baseline gap-1 font-mono text-xs">
	{#if label}<span class="text-zinc-500">{label}</span>{/if}
	<span class="font-semibold {gradeTone(grade)}">{grade}</span>
</span>
