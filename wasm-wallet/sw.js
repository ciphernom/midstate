// pool_client.bundle.js is deliberately NOT precached: it is imported lazily by
// index.html only when a miner configures a multiaddr pool, exactly as
// light_client.bundle.js is. Precaching it would push the libp2p bundle onto
// every user including solo miners who never touch it.
const CACHE_NAME = 'midstate-wallet-v99';

const ASSETS = [
  'index.html',
  'worker.js',
  'miner.js',
  // The mining kernel, fetched at runtime by gpu_miner.js. Precached because a
  // 404 here silently downgrades every GPU-capable browser to the CPU path.
  'pow.wgsl',
  'manifest.json',
  'pkg/wasm_wallet.js',
  'pkg/wasm_wallet_bg.wasm'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            return caches.delete(cache);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.origin !== location.origin) return;
  if (event.request.method !== 'GET') return;

  // 1. Network-First for HTML (Navigation requests)
  // Ensures the user always gets the freshest index.html on their first visit.
  if (event.request.mode === 'navigate' || event.request.headers.get('accept').includes('text/html')) {
    event.respondWith(
      fetch(event.request)
        .then((networkResponse) => {
          // Save the fresh HTML to the cache for offline use
          const clone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return networkResponse;
        })
        .catch(() => {
          // If the network fails (offline), fall back to the cached HTML
          return caches.match(event.request);
        })
    );
    return;
  }

  // 2. Cache-First for everything else (WASM, JS, CSS, JSON)
  // Keeps the app lightning fast and offline-capable.
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).catch(() => {
        return new Response('Offline', { status: 503, statusText: 'Offline' });
      });
    })
  );
});
