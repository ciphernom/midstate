const CACHE_NAME = 'midstate-wallet-v3';

const ASSETS = [
  'index.html',
  'worker.js',
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

  // 1. ONLY intercept requests from our own domain.
  // Bypass the Service Worker completely for external RPC node traffic.
  if (url.origin !== location.origin) {
    return;
  }

  // 2. Only cache GET requests
  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      
      // 3. Catch the fetch error so it doesn't throw a red console error
      return fetch(event.request).catch((err) => {
        console.error("Service Worker fetch failed for:", event.request.url, err);
      });
    })
  );
});
