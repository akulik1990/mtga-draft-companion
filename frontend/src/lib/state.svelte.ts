import { api, type AdvisorKey, type Snapshot } from './api';

const PREF = 'companion.advisor';

export const app = $state({
	snap: null as Snapshot | null,
	connected: false,
	error: '' as string,
	// which advisor drives highlights; UI preference, not server config
	advisor: ((typeof localStorage !== 'undefined' && localStorage.getItem(PREF)) || 'upstream') as AdvisorKey
});

export function setAdvisor(key: AdvisorKey) {
	app.advisor = key;
	localStorage.setItem(PREF, key);
}

export async function refresh() {
	try {
		app.snap = await api.state();
		app.error = '';
	} catch (e) {
		app.error = String(e);
	}
}

let socket: WebSocket | undefined;
export function connect() {
	if (socket) return;
	const proto = location.protocol === 'https:' ? 'wss' : 'ws';
	socket = new WebSocket(`${proto}://${location.host}/api/ws`);
	socket.onopen = () => (app.connected = true);
	socket.onmessage = (ev) => (app.snap = JSON.parse(ev.data));
	socket.onclose = () => {
		app.connected = false;
		socket = undefined;
		setTimeout(connect, 2000);
	};
	socket.onerror = () => socket?.close();
}
