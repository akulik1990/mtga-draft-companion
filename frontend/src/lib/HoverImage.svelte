<script lang="ts">
	import { hover } from './hover.svelte';
</script>

{#if hover.src}
	<img src={hover.src} alt="" class="pointer-events-none fixed z-50 w-64 rounded-xl shadow-2xl" style="left:{Math.max(8, hover.x)}px; top:{hover.y}px" />
{/if}
