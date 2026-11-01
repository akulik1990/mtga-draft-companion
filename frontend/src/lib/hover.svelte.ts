/** One floating card image for the whole page: call show() on mouseenter, hide() on mouseleave. */
export const hover = $state<{ src: string; x: number; y: number } | { src: ''; x: 0; y: 0 }>({ src: '', x: 0, y: 0 });

export function show(card: { image?: string[] }, e: MouseEvent) {
	const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
	// Open on the side with more room, level with the row, inside the window.
	const right = r.right + 280 < innerWidth;
	Object.assign(hover, { src: card.image?.[0] ?? '', x: right ? r.right + 8 : r.left - 272, y: Math.max(8, Math.min(r.top - 120, innerHeight - 370)) });
}
export const hide = () => Object.assign(hover, { src: '', x: 0, y: 0 });
