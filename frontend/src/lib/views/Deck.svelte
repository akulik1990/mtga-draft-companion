<script lang="ts">
	import { api, type Build, type DeckCard } from '../api';
	import { pct } from '../cards';

	let builds = $state<Build[]>([]);
	let active = $state(0);
	let busy = $state(false);
	let error = $state('');
	let copied = $state(false);
	let ran = $state(false);

	async function build() {
		busy = true; error = '';
		try { builds = await api.buildDecks(); active = 0; ran = true; } catch (e) { error = String(e); } finally { busy = false; }
	}
	const b = $derived(builds[active]);
	const count = (cs: DeckCard[]) => cs.reduce((n, c) => n + c.count, 0);
	async function copy() {
		await navigator.clipboard.writeText(b.arena_export);
		copied = true; setTimeout(() => (copied = false), 1500);
	}
</script>

<div class="mb-3 flex items-center gap-3">
	<button class="rounded bg-emerald-700 px-3 py-1 text-sm hover:bg-emerald-600 disabled:opacity-50" onclick={build} disabled={busy}>
		{busy ? 'building…' : 'Build decks'}
	</button>
	{#each builds as x, i}
		<button class="rounded px-2 py-1 text-sm {i === active ? 'bg-zinc-700' : 'text-zinc-400 hover:text-zinc-200'}" onclick={() => (active = i)}>
			{x.label}
		</button>
	{/each}
	{#if error}<span class="text-sm text-rose-400">{error}</span>{/if}
</div>

{#if b}
	<div class="grid gap-6 md:grid-cols-[1fr_1fr_1fr]">
		<section>
			<h3 class="mb-1 text-sm font-semibold text-zinc-400">Deck ({count(b.deck)}) · {b.colors.join('')}{b.record ? ` · ${b.record}` : ''}</h3>
			{#each b.deck as c}
				<div class="flex gap-2 text-sm"><span class="w-4 font-mono text-zinc-500">{c.count}</span><span class="flex-1 truncate">{c.name}</span><span class="font-mono text-xs text-zinc-500">{pct(c.gihwr)}</span></div>
			{/each}
			{#if b.note}<p class="mt-2 text-xs text-zinc-500">{b.note}</p>{/if}
		</section>
		<section>
			<h3 class="mb-1 text-sm font-semibold text-zinc-400">Sideboard ({count(b.sideboard)})</h3>
			{#each b.sideboard as c}
				<div class="flex gap-2 text-sm text-zinc-400"><span class="w-4 font-mono text-zinc-600">{c.count}</span><span class="flex-1 truncate">{c.name}</span></div>
			{/each}
		</section>
		<section>
			<h3 class="mb-1 text-sm font-semibold text-zinc-400">Simulator</h3>
			<table class="w-full font-mono text-xs"><tbody>
				{#each Object.entries(b.stats ?? {}) as [k, v]}
					<tr><td class="text-zinc-500">{k}</td><td class="text-right">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v}</td></tr>
				{/each}
			</tbody></table>
			<button class="mt-3 rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={copy}>{copied ? 'copied' : 'Copy Arena export'}</button>
			<pre class="mt-2 max-h-64 overflow-auto rounded bg-zinc-900 p-2 text-xs text-zinc-400">{b.arena_export}</pre>
		</section>
	</div>
{:else if ran && !busy}
	<p class="text-sm text-zinc-500">Upstream's builder returned no deck. It needs a near-complete pool (about 23 playable spells).</p>
{:else if !busy}
	<p class="text-sm text-zinc-500">Builds from the current pool with upstream's deck builder and mana simulator. First build compiles the simulator; expect a few seconds.</p>
{/if}
