// /videos/ (list) and /videos/?v=<id> (one video). Prerendered to videos/index.html so the same build runs under
// FastAPI and as a static site (tools/video_site.py); relative asset URLs let it sit under a Pages sub-path.
export const trailingSlash = 'always';
