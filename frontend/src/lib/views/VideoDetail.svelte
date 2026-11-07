<script lang="ts">
import { onDestroy, onMount } from 'svelte';
import { api, IDEA_STATUSES, type EngineDeck, type Idea, type IdeaStatus, type VideoData, type VideoEngine } from '../api';

	let { id }: { id: string } = $props();
	let data = $state<VideoData | null>(null);
	let error = $state('');
	let commit = $state('latest');
	let engine = $state<VideoEngine | null>(null);
	let against = $state('');
	let other = $state<VideoEngine | null>(null);
	let build = $state(0);
	let showEng = $state(true);
	let showTbl = $state(false);
	let showPacks = $state(false);
	let showTl = $state(true);
	const setTl = () => { try { localStorage.setItem('videoTimeline', showTl ? '1' : '0'); } catch { /* private mode */ } };
	const setPacks = () => { try { localStorage.setItem('videoPacks', showPacks ? '1' : '0'); } catch { /* private mode */ } };
	const setEng = () => { try { localStorage.setItem('videoEngine', showEng ? '1' : '0'); } catch { /* private mode */ } };
	let statusFilter = $state('');
	let phaseFilter = $state('');

	// ── player: the embed is a plain iframe; the IFrame API (loaded at runtime) only adds seekTo
	// svelte-ignore state_referenced_locally (the page re-creates this component per id)
	const BASE =`https://www.youtube.com/embed/${id}?enablejsapi=1`;
	let src = $state(BASE);
	let frame = $state<HTMLIFrameElement>();
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let player: any = null;
	const yt = () => (window as any).YT;
	let now = $state(-1); // current video time in seconds, -1 until the player reports one
	let timer: ReturnType<typeof setInterval> | undefined;
	const tick = () => { const t = player?.getCurrentTime?.(); if (typeof t === 'number') now = t; };
	let apiReady = $state(!!(window as any).YT?.Player);
	let attached: HTMLIFrameElement | undefined;
	function detach() { clearInterval(timer); try { player?.destroy?.(); } catch { /* iframe already gone */ } player = null; attached = undefined; }
	function attach() {
		if (!frame || !yt()?.Player || attached === frame) return;
		if (player) detach();
		attached = frame;
		player = new (yt().Player)(frame, {
			events: {
				onReady: () => { tick(); if (player?.getPlayerState?.() === 1) { clearInterval(timer); timer = setInterval(tick, 500); } },
				// state 1 = playing: poll every 500 ms; any other state: stop, one last read so seeks while paused show
				onStateChange: (ev: { data: number }) => {
					clearInterval(timer);
					tick();
					if (ev.data === 1) timer = setInterval(tick, 500);
				},
			},
		});
	}
	// the iframe only exists once data has loaded, so attach reactively (api ready AND frame present)
	$effect(() => { if (frame && apiReady) attach(); });
	onDestroy(detach);
	function seek(t: string | number) {
		const s = secs(t);
		now = s;
		if (player?.seekTo) { player.seekTo(s, true); player.playVideo?.(); }
		else src = `${BASE}&start=${s}&autoplay=1`; // API unavailable: reload the iframe at the time
		kick++; pausedUntil = 0;
		playerBox?.scrollIntoView({ block: 'start', behavior: 'smooth' });
	}
	function secs(t: string | number) {
		if (typeof t === 'number') return Math.floor(t);
		return String(t).split(/[-–]/)[0].trim().split(':').reduce((a, p) => a * 60 + (parseInt(p) || 0), 0);
	}
	// display only: drop verbatim quoted speech (a quote opens at a word boundary, closes with the same mark before punctuation/end)
	const unquote = (t: string | undefined) => {
		let cut = false;
		const out = (t ?? '').replace(/(^|\W)(['"])(.+?)\2(?=[\s.,;:!?)]|$)/g, (_, pre, _q, body) => (body.trim().split(/\s+/).length >= 4 ? ((cut = true), pre) : pre + body)).replace(/\s+/g, ' ').trim();
		return cut ? out.replace(/([;,:]\s*){2,}/g, '; ').replace(/^[\s;:,.\-]+|[\s;:,.\-]+$/g, '').replace(/\s+([.,;:])/g, '$1') : out;
	};
	const mss = (t: string | number) => { const s = secs(t); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

	onMount(async () => {
		try { showEng = localStorage.getItem('videoEngine') !== '0'; } catch { /* default on */ }
		try { showPacks = localStorage.getItem('videoPacks') === '1'; } catch { /* default closed */ }
		try { showTl = localStorage.getItem('videoTimeline') !== '0'; } catch { /* default shown */ }
		try { showTbl = localStorage.getItem('videoPicksTable') === '1'; } catch { /* default closed */ }
		if (!apiReady) {
			const prev = (window as any).onYouTubeIframeAPIReady;
			(window as any).onYouTubeIframeAPIReady = () => { prev?.(); apiReady = true; };
			const s = document.createElement('script');
			s.src = 'https://www.youtube.com/iframe_api';
			document.head.appendChild(s);
		}
		try { data = await api.video(id); engine = data.engine; } catch (e) { error = String(e); }
	});

	async function pick(which: 'engine' | 'other', c: string) {
		try {
			const e = c === 'latest' ? data!.engine : c ? await api.videoEngine(id, c) : null;
			if (which === 'engine') engine = e; else other = e;
		} catch (err) { error = String(err); }
	}

	// ── picks
	const key = (p: { pack: number; pick: number }) => `${p.pack}-${p.pick}`;
	const eMap = $derived(new Map((engine?.picks ?? []).map((p) => [key(p), p])));
	const oMap = $derived(new Map((other?.picks ?? []).map((p) => [key(p), p])));
	const rows = $derived((data?.picks.picks ?? []).map((p) => ({ p, e: eMap.get(key(p)), o: oMap.get(key(p)) })));

	// ── timeline: last pick whose timestamp is <= now (rows without a timestamp are skipped, so the nearest earlier one stays lit)
	const active = $derived(rows.reduce((a, { p }, i) => (p.time && secs(p.time) <= now ? i : a), -1));
	let box = $state<HTMLElement>();
	let playerBox = $state<HTMLElement>();
	let col = $state<HTMLElement>();
	let hd = $state<HTMLElement>();
	let bar = $state<HTMLElement>();
	let tl = $state(0); // desktop timeline box height in px: one tallest row, capped at 30vh
	let top = $state(0); // px above the player: page offset + header + gap
	function measure() {
		if (!col || !hd || !bar) return;
		const h = Math.max(0, ...[...(box?.querySelectorAll<HTMLElement>('[data-i]') ?? [])].map((r) => r.offsetHeight));
		tl = showTl && box ? Math.floor(Math.min(h + 2, innerHeight * 0.3)) : 0; // +2 = box border
		top = Math.ceil(col.getBoundingClientRect().top + scrollY + hd.offsetHeight + 12 + bar.offsetHeight + 4); // header + gap + toggle bar + gap
	}
	$effect(() => { // re-measure on load, resize, rewrap, toggles (rows/showEng/showPacks are read so the observer is rebuilt)
		rows; showEng; showPacks; showTl;
		if (!hd || !bar) return;
		const ro = new ResizeObserver(measure);
		for (const el of [hd, bar, ...(box ? [box, ...box.querySelectorAll('[data-i]')] : [])]) ro.observe(el);
		measure();
		return () => ro.disconnect();
	});
	let kick = $state(0); // bumped on a click so the box re-aligns even when the active row did not change
	let pausedUntil = 0; // set by the user's own scrolling of the box
	const hold = () => { pausedUntil = Date.now() + 4000; };
	$effect(() => {
		kick;
		if (Date.now() < pausedUntil) return;
		const row = active >= 0 ? box?.querySelector<HTMLElement>(`[data-i="${active}"]`) : null;
		if (box && row) box.scrollTo({ top: row.offsetTop, behavior: 'smooth' }); // own box only, never the page
	});

	// ── colour path: the latest stated plan at each pick next to v2's top lane after the pick
	const plans = $derived.by(() => {
		const at = new Map((data?.picks.colour_path ?? []).map((c) => [c.after, c]));
		let cur = '';
		return rows.map(({ p }) => {
			const c = at.get(`P${p.pack}p${p.pick}`);
			const changed = !!c && c.main !== cur;
			if (c) cur = c.main;
			return { cur, changed };
		});
	});

	// ── decks
	const hisDeck = $derived(data?.deck ? Object.entries({ ...data.deck.main, ...data.deck.lands }).map(([name, count]) => ({ name, count })) : []);
	const ours = $derived<EngineDeck | undefined>(engine?.builder[build]);
	const names = (cs: { name: string }[]) => new Set(cs.map((c) => c.name));
	const BASICS = ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest'];
	const order = (cs: { name: string; count: number }[]) =>
		[...cs].sort((a, b) => Number(BASICS.includes(a.name)) - Number(BASICS.includes(b.name)) || a.name.localeCompare(b.name));

	// ── ideas
	const nick = $derived(data?.video.player || 'Player');
	const ideas = $derived(data?.ideas ?? []);
	const phases = $derived([...new Set(ideas.map((i) => i.phase).filter(Boolean))]);
	const stateOf = (i: Idea): IdeaStatus => data?.status[i.id] ?? { status: 'untested', result: '', doc: '' };
	const shown = $derived(ideas.filter((i) => (!statusFilter || stateOf(i).status === statusFilter) && (!phaseFilter || i.phase === phaseFilter)));
	async function setStatus(i: Idea, status: string) {
		try {
			const s = stateOf(i);
			data!.status[i.id] = await api.setIdea(id, i.id, { ...s, status });
		} catch (e) { error = String(e); }
	}
	const lanePct = (v: number) => `${Math.round(v * 100)}%`;
</script>

{#if error}<p class="mb-3 text-sm text-rose-400">{error}</p>{/if}
{#if !data}
	<p class="text-zinc-500">loading…</p>
{:else}
	<div bind:this={col} class="col" style={top ? `--tl: ${tl}px; --top: ${top}px` : ''}>
	<div bind:this={hd} class="mb-3 flex justify-center flex-wrap items-baseline gap-x-4 gap-y-1 text-sm">
		<a href="." class="text-zinc-400 hover:text-zinc-100">&larr; all videos</a>
		<h1 class="text-lg font-semibold">{data.video.title}</h1>
		<span class="text-zinc-400">{data.video.player} · {data.video.set}</span>
		<label class="text-zinc-300"><input type="checkbox" bind:checked={showEng} onchange={setEng} /> Engine output</label>
		<label class="text-zinc-300"><input type="checkbox" bind:checked={showPacks} onchange={setPacks} /> Packs</label>
{#if engine && showEng}<span class="text-zinc-500">v2 {engine.commit} ({engine.date}): top-1 {engine.top1}/{engine.n}, top-3 {engine.top3}/{engine.n}</span>{/if}
	</div>

	<section class="mb-6">
		<div bind:this={playerBox} class="pw aspect-video w-full">
			<iframe bind:this={frame} class="h-full w-full rounded" {src} title={data.video.title} allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>
		</div>
		<div bind:this={bar} class="mb-1 text-xs"><button type="button" class="rounded px-1 py-0.5 text-zinc-300 hover:text-zinc-100" onclick={() => { showTl = !showTl; setTl(); }}>Timeline {showTl ? '▾' : '▸'}</button></div>
		{#if showTl}
		<div bind:this={box} role="list" onwheel={hold} ontouchmove={hold} class="tl relative max-h-[40vh] overflow-y-auto overflow-x-hidden rounded border border-zinc-800 text-xs">
			{#each rows as { p, e }, i (key(p))}
				{@const miss = showEng && !!e && e.top3[0]?.name !== p.taken}
{@const inTop = !!e && e.top3.some((c) => c.name === p.taken)}
				{@const lanes = e ? Object.entries(e.lanes_after).slice(0, 3) : []}
				<div data-i={i} role="presentation" onclick={() => p.time && seek(p.time)} class="{p.time ? 'cursor-pointer ' : ''}border-t border-zinc-900 px-2 py-1.5 first:border-t-0 {i === active ? 'bg-sky-950 ring-1 ring-inset ring-sky-600' : miss ? 'bg-amber-950/30' : ''}">
					<div class="flex flex-wrap items-baseline gap-x-2">
						{#if p.time}<span class="text-sky-400 tabular-nums">{mss(p.time)}</span>{/if}
						<span class="text-zinc-500">P{p.pack}p{p.pick}</span>
						<span class="text-sm font-medium break-words">{p.taken}</span>
						{#if showEng && e && e.scored !== false}<span class={miss ? 'text-amber-400' : 'text-emerald-400'}>{miss ? `v2 ranked it #${e.his_rank ?? '?'}` : 'v2 agrees ✓'}</span>{/if}
					</div>
					<div class="mt-1 lg:flex lg:items-start lg:gap-6">
					<div class="border-l-2 border-sky-700 pl-2 text-zinc-300 {showEng ? 'lg:w-[55%]' : 'lg:w-full'}">
						<span class="font-semibold text-sky-400">{nick}</span>
						{#if unquote(p.plan)}<div class="break-words"><span class="text-zinc-500">plan:</span> {unquote(p.plan)}</div>{/if}
						{#if unquote(p.reasoning)}<div class="break-words">{unquote(p.reasoning)}</div>{/if}
						{#if p.considered && Object.keys(p.considered).length}
							{@const cons = Object.entries(p.considered)}
							{@const inline = cons.length <= 3 && cons.every(([c, w]) => c.length + unquote(w).length <= 40)}
							{#if inline}
								<div class="break-words"><span class="text-zinc-500">considered:</span> {#each cons as [card, why], j}<span class="mr-2 inline-block">{#if j}<span class="mr-2 text-zinc-600">·</span>{/if}<span class="text-zinc-100">{card}</span>{unquote(why) ? `: ${unquote(why)}` : ''}</span>{/each}</div>
							{:else}
								<div class="text-zinc-500">considered:</div>
								{#each cons as [card, why]}<div class="break-words"><span class="text-zinc-100">{card}</span>{unquote(why) ? `: ${unquote(why)}` : ''}</div>{/each}
							{/if}
						{/if}
					</div>
					{#if showEng}
<div class="mt-1 border-l-2 border-amber-600 pl-2 text-zinc-400 lg:mt-0 lg:w-[45%]">
						<span class="font-semibold text-amber-400">ENGINE</span>
						{#if !e || e.scored === false}
							<div>engine: not scored</div>
						{:else}
							<div class="break-words">{#each e.top3 as c, k}<span class="mr-2 inline-block">{#if k}<span class="mr-2 text-zinc-600">·</span>{/if}<span class={c.name === p.taken ? 'font-semibold text-emerald-400' : ''}>{k + 1}. {c.name} <span class="text-zinc-500 tabular-nums">{c.score.toFixed(1)}</span></span></span>{/each}</div>
							{#if !inTop}<div class="break-words text-emerald-400">their card: rank {e.his_rank ?? '?'}, score {e.his_score?.toFixed(1) ?? '?'}</div>{/if}
							{#if miss && e.his_score != null && e.top3[0]}<div class="text-amber-400">gap to #1: −{(e.top3[0].score - e.his_score).toFixed(1)}</div>{/if}
							{#if e.reasons_top.length}<div class="break-words"><span class="text-zinc-500">why #1:</span> {e.reasons_top.join(' · ')}</div>{/if}
							{#if miss && e.reasons_his.length}<div class="break-words"><span class="text-zinc-500">why their card:</span> {e.reasons_his.join(' · ')}</div>{/if}
							{#if lanes.length}<div><span class="text-zinc-500">lanes after the pick:</span> {lanes.map(([l, v]) => `${l} ${lanePct(v)}`).join(' · ')}</div>{/if}
						{/if}
</div>
{/if}
					</div>
{#if showPacks && p.pack_cards?.length}
						<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
						<details class="mt-1 text-zinc-500" open onclick={(ev) => ev.stopPropagation()}>
							<summary class="cursor-pointer">pack ({p.pack_cards.length})</summary>
							<div class="break-words">{p.pack_cards.join(', ')}</div>
						</details>
					{/if}
				</div>
			{:else}
				<p class="p-2 text-zinc-500">no picks recorded</p>
			{/each}
			<div class="h-[calc(40vh-4rem)] lg:h-[20vh]" aria-hidden="true"></div> <!-- lets the last rows reach the top of the box -->
		</div>
		{/if}
	</section>
	</div>

	<details class="mb-6" open={showTbl} ontoggle={(ev) => { showTbl = ev.currentTarget.open; try { localStorage.setItem('videoPicksTable', showTbl ? '1' : '0'); } catch { /* private mode */ } }}>
	<summary class="mb-2 cursor-pointer text-sm font-semibold text-zinc-200">Picks table</summary>
	<section class="mb-6">
		<div class="mb-2 flex flex-wrap items-center gap-4 text-xs text-zinc-500">
			<h2 class="text-sm font-semibold text-zinc-200">Picks</h2>
			{#if showEng}<label>engine
<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" bind:value={commit} onchange={() => pick('engine', commit)}>
					<option value="latest">latest{data.engine ? ` (${data.engine.commit})` : ''}</option>
					{#each data.commits as c}<option value={c}>{c}</option>{/each}
				</select>
			</label>
			<label>compare with
				<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" bind:value={against} onchange={() => pick('other', against)}>
					<option value="">none</option>
					{#each data.commits as c}<option value={c}>{c}</option>{/each}
				</select>
			</label>
			{#if other}<span>{rows.filter((r) => r.e && r.o && r.e.top3[0]?.name !== r.o.top3[0]?.name).length} picks where v2's #1 changed</span>{/if}{/if}
		</div>
		<div class="overflow-x-auto">
			<table class="w-full min-w-[80rem] text-sm">
				<thead class="text-left text-xs text-zinc-500">
					<tr>
						<th class="pr-3 font-normal">Time</th><th class="pr-3 font-normal">Pick</th><th class="pr-3 font-normal">{nick} picked</th>
						{#if showEng}<th class="pr-3 font-normal">v2 #1</th><th class="pr-3 font-normal">v2 #2</th><th class="pr-3 font-normal">v2 #3</th>{/if}
						<th class="pr-3 font-normal">{nick} said</th>{#if showEng}<th class="font-normal">Our reasoning</th>{/if}
					</tr>
				</thead>
				<tbody>
					{#each rows as { p, e, o } (key(p))}
						{@const miss = showEng && !!e && e.top3[0]?.name !== p.taken}
<tr onclick={() => p.time && seek(p.time)} class="{p.time ? 'cursor-pointer ' : ''}border-t border-zinc-800 align-top {miss ? 'bg-amber-950/30' : ''}">
							<td class="py-1 pr-3"><button class="text-sky-400 tabular-nums hover:underline" title={p.time}>{mss(p.time)}</button></td>
							<td class="pr-3 whitespace-nowrap">P{p.pack}p{p.pick}</td>
							<td class="pr-3 font-medium">{p.taken}</td>
							{#if showEng}{#each [0, 1, 2] as k}
								{@const c = e?.top3[k]}
								<td class="pr-3 {c?.name === p.taken ? 'font-semibold text-emerald-400' : ''}">
									{#if c}
										{c.name} <span class="text-xs text-zinc-500 tabular-nums">{c.score.toFixed(1)}</span>{c.name === p.taken ? ' ✓' : ''}
										{#if k === 0 && o && o.top3[0]?.name !== c.name}<div class="text-xs text-amber-400">was {o.top3[0]?.name}</div>{/if}
									{/if}
								</td>
							{/each}{/if}
<td class="max-w-sm pr-3 text-xs text-zinc-300">{unquote(p.reasoning)}</td>
							{#if showEng}<td class="max-w-sm text-xs text-zinc-400">
								{#if e}
									{#if !miss}
										{e.reasons_top.join(' · ')}
									{:else}
										<div>#1: {e.reasons_top.join(' · ')}</div>
										<div class="mt-1 text-zinc-500">{nick}{e.his_rank ? ` (rank ${e.his_rank}, ${e.his_score?.toFixed(1)})` : ''}: {e.reasons_his.join(' · ')}</div>
									{/if}
								{/if}
</td>{/if}
</tr>
{/each}
</tbody>
			</table>
		</div>
	</section>

	<section class="mb-6">
		<h2 class="mb-2 text-sm font-semibold text-zinc-200">Colour plan by pick</h2>
		<div class="max-h-[28rem] overflow-auto text-xs">
			<table class="w-full">
				<thead class="sticky top-0 bg-zinc-950 text-left text-zinc-500">
					<tr><th class="font-normal">Pick</th><th class="font-normal">{nick} plan</th>{#if showEng}<th class="font-normal">v2 top lane after the pick</th>{/if}</tr>
				</thead>
				<tbody>
					{#each rows as { p, e }, i (key(p))}
						{@const top = e && Object.entries(e.lanes_after)[0]}
						<tr class="border-t border-zinc-900 {plans[i].changed ? 'text-zinc-100' : 'text-zinc-500'}">
							<td class="pr-2 whitespace-nowrap">P{p.pack}p{p.pick}</td>
							<td class="pr-2">{plans[i].changed ? plans[i].cur : '·'}</td>
							{#if showEng}<td class="whitespace-nowrap {top && plans[i].cur && !top[0].split('').some((c) => plans[i].cur.includes(c)) ? 'text-amber-400' : ''}">{top ? `${top[0]} ${lanePct(top[1])}` : '-'}</td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>
	</details>

	<section class="mb-6">
		<div class="mb-2 flex items-center gap-4 text-xs text-zinc-500">
			<h2 class="text-sm font-semibold text-zinc-200">Deck</h2>
			{#if showEng && engine && engine.builder.length > 1}
				<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" bind:value={build}>
					{#each engine.builder as b, i}<option value={i}>{b.label ?? `build ${i + 1}`}</option>{/each}
				</select>
			{/if}
		</div>
		<div class="grid gap-6 md:grid-cols-2">
			{#each [{ title: `${nick}'s 40${showEng && engine?.his_deck?.deck_wr ? ` · deck model ${engine.his_deck.deck_wr}` : ''}`, cards: hisDeck, vs: ours?.cards ?? [] }, { title: `Our builder's 40 from their pool${ours ? ` · ${ours.main ?? ''}${ours.splash ? ' + ' + ours.splash : ''} · deck model ${ours.deck_wr}` : ''}`, cards: ours?.cards ?? [], vs: hisDeck }].slice(0, showEng ? 2 : 1) as d}
				{@const mine = names(d.vs)}
				<div>
					<h3 class="mb-1 text-xs text-zinc-500">{d.title}</h3>
					<ul class="text-sm">
						{#each order(d.cards) as c (c.name)}
							<li class={!showEng || mine.has(c.name) || BASICS.includes(c.name) ? '' : 'text-amber-400'}>{c.count} {c.name}</li>
						{:else}
							<li class="text-zinc-500">none</li>
						{/each}
					</ul>
				</div>
			{/each}
		</div>
		{#if showEng}<p class="mt-1 text-xs text-zinc-500">Amber: not in the other deck.</p>{/if}
	</section>

	{#if showEng}
	<section>
		<div class="mb-2 flex flex-wrap items-center gap-4 text-xs text-zinc-500">
			<h2 class="text-sm font-semibold text-zinc-200">Ideas ({shown.length}/{ideas.length})</h2>
			<label>status
				<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" bind:value={statusFilter}>
					<option value="">all</option>{#each IDEA_STATUSES as s}<option value={s}>{s}</option>{/each}
				</select>
			</label>
			<label>phase
				<select class="rounded bg-zinc-900 px-1 py-0.5 text-zinc-200" bind:value={phaseFilter}>
					<option value="">all</option>{#each phases as ph}<option value={ph}>{ph}</option>{/each}
				</select>
			</label>
		</div>
		{#if !ideas.length}<p class="text-sm text-zinc-500">No ideas recorded for this video yet.</p>{/if}
		<table class="w-full text-sm">
			<thead class="text-left text-xs text-zinc-500">
				<tr><th class="pr-3 font-normal">Time</th><th class="pr-3 font-normal">Phase</th><th class="pr-3 font-normal">Claim</th><th class="pr-3 font-normal">Status</th><th class="pr-3 font-normal">Result</th><th class="font-normal">Doc</th></tr>
			</thead>
			<tbody>
				{#each shown as i (i.id)}
					{@const s = stateOf(i)}
					<tr class="border-t border-zinc-800 align-top">
						<td class="py-1 pr-3"><button class="text-sky-400 tabular-nums hover:underline" onclick={() => seek(i.time)}>{mss(i.time)}</button></td>
						<td class="pr-3 whitespace-nowrap">{i.phase}</td>
						<td class="max-w-xl pr-3" title={i.quote}>{i.claim}</td>
						<td class="pr-3">
							<select class="rounded bg-zinc-900 px-1 py-0.5" disabled={data.readonly} value={s.status} onchange={(e) => setStatus(i, e.currentTarget.value)}>
								{#each IDEA_STATUSES as st}<option value={st}>{st}</option>{/each}
							</select>
						</td>
						<td class="max-w-xs pr-3 text-xs text-zinc-300">{s.result}</td>
						<td class="text-xs">{#if /^https?:/.test(s.doc)}<a class="text-sky-400 hover:underline" href={s.doc} target="_blank" rel="noreferrer">{s.doc}</a>{:else}<span class="text-zinc-400">{s.doc}</span>{/if}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</section>
	{/if}
{/if}

<style>
	/* desktop: timeline is full width; player is centred, width capped so header + player + one-row timeline fit the viewport (--top and --tl are measured in measure()) */
	@media (min-width: 1024px) {
		.col {
			container-type: inline-size;
			--pw: min(100cqw, calc((100vh - var(--top, 5rem) - var(--tl, 34vh)) * 16 / 9));
		}
		.pw {
			margin-inline: auto;
			max-width: var(--pw);
			transition: max-width 0.3s;
		}
		.col .tl {
			max-height: none;
			height: var(--tl, 34vh);
		}
	}
</style>
