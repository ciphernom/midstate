const CACHE_NAME = 'midstate-wallet-v7';

const ASSETS = [
  'index.html',
  'worker.js',
  'miner.js',
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
