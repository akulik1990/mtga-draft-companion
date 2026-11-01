<script lang="ts">
	import type { Card, Rec } from './api';
	import { pct } from './cards';

	let { card, archetype, upstream, v2, onclose, onpick, picking = false }: { card: Card; archetype: string; upstream?: Rec; v2?: Rec; onclose: () => void; onpick?: (c: Card) => void; picking?: boolean } = $props();

	const rows = $derived(
		Object.entries(card.deck_colors ?? {})
			.filter(([, s]) => s.samples > 0 && s.gihwr > 0)
			.sort((a, b) => (a[0] === 'All Decks' ? -1 : b[0] === 'All Decks' ? 1 : b[1].gihwr - a[1].gihwr))
	);
	const cols = ['gihwr', 'ohwr', 'gpwr', 'gdwr', 'gnswr', 'iwd', 'alsa', 'ata', 'samples'] as const;
</script>

<svelte:window onkeydown={(e) => e.key === 'Escape' && onclose()} />

{#snippet adv(k: string, r?: Rec)}
	{#if r}
		<p class="mt-2"><span class="font-mono text-zinc-500">{k}</span> #{r.rank} · {r.contextual_score.toFixed(1)}
			{#if r.reasoning?.length}<span class="text-zinc-400">· {r.reasoning.join('; ')}</span>{/if}</p>
	{/if}
{/snippet}

<div class="fixed inset-0 z-20 flex items-center justify-center p-4">
	<button type="button" class="absolute inset-0 bg-black/80" aria-label="close" onclick={onclose}></button>
	<div role="dialog" aria-modal="true" class="relative flex max-h-full w-full max-w-5xl gap-4 overflow-auto rounded-xl bg-zinc-900 p-4">
		<img src={card.image?.[0]} alt={card.name} class="h-[480px] shrink-0 rounded-lg" />
		<div class="min-w-0 flex-1 text-sm">
			<div class="flex items-baseline justify-between gap-3">
				<h2 class="text-lg font-semibold">{card.name} <span class="font-mono text-zinc-500">{card.mana_cost}</span></h2>
				<span class="flex items-center gap-3">
					{#if onpick}<button class="rounded bg-emerald-700 px-3 py-1 text-sm hover:bg-emerald-600 disabled:opacity-50" onclick={() => onpick(card)} disabled={picking}>Pick</button>{/if}
					<button class="text-zinc-500 hover:text-zinc-200" onclick={onclose}>✕</button>
				</span>
			</div>
			<p class="text-zinc-500">{card.types?.join(' ')} · {card.rarity} {#if card.tags?.length}· {card.tags.join(', ')}{/if}</p>
			{@render adv('upstream', upstream)}
			{@render adv('v2', v2)}
			<table class="mt-3 w-full font-mono text-xs [&_td]:px-2 [&_th]:px-2">
				<thead class="text-zinc-500"><tr><th class="text-left">lane</th>{#each cols as c}<th class="text-right">{c}</th>{/each}</tr></thead>
				<tbody>
					{#each rows as [lane, s]}
						<tr class={lane === archetype ? 'bg-emerald-900/30' : lane === 'All Decks' ? 'bg-zinc-800/60' : ''}>
							<td class="py-0.5">{lane}</td>
							{#each cols as c}
								<td class="text-right">{c === 'samples' ? s[c] : c === 'alsa' || c === 'ata' ? (s[c] ?? 0).toFixed(2) : pct(s[c])}</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
</div>
