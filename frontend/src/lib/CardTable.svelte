<script lang="ts">
	import type { Card } from './api';
	import { stripe, type Column } from './format';
	import Pips from './Pips.svelte';

	let { cards, columns, visible, highlight = '', onrow }: {
		cards: Card[]; columns: Column[]; visible: string[]; highlight?: string; onrow: (c: Card) => void;
	} = $props();

	let sortKey = $state('');
	let asc = $state(true);
	const shown = $derived(columns.filter((c) => visible.includes(c.key)));
	const rows = $derived.by(() => {
		const col = columns.find((c) => c.key === sortKey);
		if (!col) return cards;
		return [...cards].sort((a, b) => (asc ? 1 : -1) * (col.sort(a) - col.sort(b)));
	});
	function by(key: string) {
		if (sortKey === key) asc = !asc;
		else { sortKey = key; asc = true; }
	}
</script>

<div class="overflow-x-auto">
	<table class="w-full font-mono text-sm">
		<thead class="text-zinc-500">
			<tr>
				<th class="px-2 py-1 text-left font-normal"><button class="hover:text-zinc-200" onclick={() => (sortKey = '')}>card</button></th>
				{#each shown as c (c.key)}
					<th class="px-2 py-1 text-right font-normal" title={c.title}>
						<button class="hover:text-zinc-200 {sortKey === c.key ? 'text-zinc-200' : ''}" onclick={() => by(c.key)}>{c.label}{sortKey === c.key ? (asc ? ' ▲' : ' ▼') : ''}</button>
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each rows as card (card.name)}
				<tr class="cursor-pointer border-l-4 {stripe(card)} {card.name === highlight ? 'bg-emerald-900/30' : 'hover:bg-zinc-800/60'}" onclick={() => onrow(card)}>
					<td class="px-2 py-1.5 font-sans text-base">{card.name}</td>
					{#each shown as c (c.key)}
						<td class="px-2 py-1.5 text-right {c.cls?.(card) ?? 'text-zinc-300'}">{#if c.kind === 'colors'}<Pips colors={card.colors ?? []} size="h-4 w-4" />{:else}{c.show(card)}{/if}</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>
