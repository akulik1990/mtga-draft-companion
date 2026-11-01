<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type Datasets, type Settings } from '../api';
	import { app, setAdvisor, refresh } from '../state.svelte';

	let ds = $state<Datasets | null>(null);
	let st = $state<Settings | null>(null);
	let logPath = $state('');
	let msg = $state('');
	let busy = $state(false);

	onMount(async () => {
		[ds, st] = await Promise.all([api.datasets(), api.settings()]);
		logPath = st.arena_log_location;
	});
	async function run(label: string, fn: () => Promise<unknown>) {
		busy = true; msg = label + '…';
		try { await fn(); msg = label + ' done'; await refresh(); } catch (e) { msg = String(e); } finally { busy = false; }
	}
	const select = (set: string, format: string, user_group: string) =>
		run(`select ${set} ${format}`, async () => (ds = await api.selectDataset(set, format, user_group)));
	const sync = () => run('sync', async () => { const r = await api.syncDatasets(); ds = r; msg = r.messages.join(' · ') || 'up to date'; });
	const saveLog = () => run('save log path', async () => { st = await api.updateSettings({ arena_log_location: logPath }); });
	const toggleAuto = () => run('auto-sync', async () => { st = await api.updateSettings({ auto_sync_datasets: !st!.auto_sync_datasets }); });
</script>

<div class="grid max-w-4xl gap-6">
	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Highlights follow</h3>
		{#each ['upstream', 'v2'] as k}
			<label class="mr-4 text-sm"><input type="radio" name="adv" checked={app.advisor === k} onchange={() => setAdvisor(k as 'upstream' | 'v2')} /> {k}</label>
		{/each}
	</section>

	<section>
		<h3 class="mb-2 text-sm font-semibold text-zinc-400">Player.log</h3>
		<p class="mb-1 text-xs text-zinc-500">active: <span class="font-mono">{st?.active_log ?? '…'}</span>{st?.arena_log_pinned ? ' (pinned)' : ' (auto-discovered)'}</p>
		<div class="flex gap-2">
			<input class="flex-1 rounded bg-zinc-900 px-2 py-1 font-mono text-sm" placeholder="leave empty for auto-discovery" bind:value={logPath} />
			<button class="rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={saveLog} disabled={busy}>Save</button>
		</div>
		<div class="mt-2 flex gap-2 text-sm">
			<button class="rounded bg-zinc-800 px-3 py-1 hover:bg-zinc-700" onclick={() => run('rescan', api.rescan)} disabled={busy}>Rescan log</button>
			<button class="rounded bg-zinc-800 px-3 py-1 hover:bg-zinc-700" onclick={() => run('reset', api.reset)} disabled={busy}>Reset draft</button>
		</div>
	</section>

	<section>
		<div class="mb-2 flex items-center gap-3">
			<h3 class="text-sm font-semibold text-zinc-400">Datasets</h3>
			<button class="rounded bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600" onclick={sync} disabled={busy}>Sync now</button>
			{#if st}<label class="text-sm"><input type="checkbox" checked={st.auto_sync_datasets} onchange={toggleAuto} /> sync at startup</label>{/if}
		</div>
		<table class="w-full text-sm">
			<thead class="text-left text-xs text-zinc-500"><tr><th>set</th><th>format</th><th>group</th><th>period</th><th class="text-right">games</th><th></th></tr></thead>
			<tbody>
				{#each ds?.datasets ?? [] as d}
					<tr class={d.file === ds?.active ? 'bg-emerald-900/30' : ''}>
						<td>{d.set}</td><td>{d.format}</td><td>{d.user_group}</td>
						<td class="font-mono text-xs text-zinc-500">{d.start_date} → {d.end_date}</td>
						<td class="text-right font-mono text-xs">{d.games?.toLocaleString?.() ?? d.games}</td>
						<td class="text-right"><button class="text-xs text-sky-300 hover:underline" onclick={() => select(d.set, d.format, d.user_group)} disabled={busy}>use</button></td>
					</tr>
				{:else}
					<tr><td colspan="6" class="text-zinc-600">no local datasets</td></tr>
				{/each}
			</tbody>
		</table>
	</section>

	{#if msg}<p class="text-sm text-zinc-400">{msg}</p>{/if}
</div>
