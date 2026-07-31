/**
 * # Midstate Serverless Phonebook (Seed Registry)
 *
 * A highly-available, stateless bootstrap registry running on Cloudflare Workers.
 * Replaces hardcoded bootstrap VPS nodes, ensuring the network can organically
 * sustain its own discovery graph.
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Route: GET /peers
    if (request.method === "GET" && url.pathname === "/peers") {
      return await handleGetPeers(env);
    }

    // Route: POST /announce
    if (request.method === "POST" && url.pathname === "/announce") {
      return await handleAnnounce(request, env);
    }

    return new Response("Not Found", { status: 404 });
  }
};

/**
 * Handles requests from fresh nodes seeking bootstrap peers.
 *
 * # Reasoning
 * Returns a randomized subset of active network peers. Shuffling is critical
 * to prevent network graph centralization (e.g., all new nodes connecting to 
 * the exact same 5 peers) and to distribute inbound connection load evenly
 * across the public nodes.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:  Registry is accessible
 * Post: Returns a randomly sampled array of up to 50 multiaddrs.
 * ```
 *
 * ```zed
 *     GetPeers
 *     --------
 *     ΞRegistry
 *     peers! : seq String
 *
 *     let AllPeers = ⋃ { addrs | (ip, addrs) ∈ Registry }
 *     pre  true
 *     post peers! ⊆ AllPeers
 *     post #peers! ≤ 50
 *     post peers! is randomly sampled
 * ```
 *
 * # Safety / Invariants
 * - **Bounded Payload:** Strictly truncates the result to 50 peers. Prevents
 *   massive JSON payloads from causing OOM crashes on embedded hardware nodes.
 */
async function handleGetPeers(env) {
  try {
    // Cloudflare KV list() returns up to 1000 keys by default. 
    // This is sufficient for the bootstrap phase of the network.
    const keys = await env.PEERS_KV.list();
    let allPeers = [];
    
    // Fetch the Multiaddrs stored under each IP address
    for (const key of keys.keys) {
      const addrs = await env.PEERS_KV.get(key.name, "json");
      if (addrs && Array.isArray(addrs)) {
        allPeers.push(...addrs);
      }
    }

    // Randomize (Fisher-Yates style shuffle via sort) and slice
    const selectedPeers = allPeers
      .sort(() => 0.5 - Math.random())
      .slice(0, 50);

    return new Response(JSON.stringify(selectedPeers), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*" // Allow browser/light clients
      }
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: "Internal Server Error" }), { 
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}

/**
 * Handles announcements from publicly routable nodes.
 *
 * # Reasoning
 * Allows the network to map itself dynamically. To prevent a malicious actor
 * from flooding the registry with millions of dead or malicious multiaddrs 
 * (Eclipse Attack vector), the KV store is keyed by the connecting IP address.
 * 
 * Furthermore, an automatic Time-To-Live (TTL) ensures nodes that crash or 
 * go offline are pruned from the registry without requiring a cleanup cron job.
 *
 * # Formal Specification
 *
 * ```text
 * Pre:
 *   - Request body is valid JSON containing an "addresses" array.
 *   - Array length <= 20.
 *
 * Post:
 *   - The caller's IP is mapped to their addresses in the KV store.
 *   - The entry is given a strict 3600-second (1 hour) expiration.
 * ```
 *
 * ```zed
 *     AnnouncePeer
 *     ------------
 *     ΔRegistry
 *     req_ip? : IPAddress
 *     addrs?  : seq String
 *
 *     pre  #addrs? > 0 ∧ #addrs? ≤ 20
 *     post Registry' = Registry ⊕ {req_ip? ↦ (addrs?, now + 3600s)}
 * ```
 *
 * # Safety / Invariants
 * - **Sybil Resistance:** Mapping by `CF-Connecting-IP` guarantees one IP 
 *   can only occupy one slot in the registry, completely neutralizing standard 
 *   spam loops.
 * - **Auto-Pruning:** Cloudflare KV `expirationTtl` guarantees state does 
 *   not leak or grow infinitely over time.
 * - **Payload Limits:** Rejects payloads requesting to store > 20 addresses
 *   to prevent KV storage bloat.
 */
async function handleAnnounce(request, env) {
  try {
    const body = await request.json();
    
    if (!body.addresses || !Array.isArray(body.addresses)) {
      return new Response("Bad Request: Expected 'addresses' array", { status: 400 });
    }

    if (body.addresses.length === 0 || body.addresses.length > 20) {
      return new Response("Bad Request: Array must contain 1-20 addresses", { status: 400 });
    }

    // Enforce string type and maximum string length for multiaddrs
    const cleanAddrs = body.addresses.filter(addr => 
      typeof addr === 'string' && addr.length < 200
    );

    // Identify the caller by their true IP
    const clientIp = request.headers.get("CF-Connecting-IP");
    if (!clientIp) {
      return new Response("Unauthorized: Missing IP", { status: 401 });
    }
    
    // Store in KV with a 1-hour expiration (3600 seconds)
    await env.PEERS_KV.put(clientIp, JSON.stringify(cleanAddrs), { expirationTtl: 3600 });

    return new Response(JSON.stringify({ status: "ok" }), { 
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: "Invalid JSON payload" }), { 
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }
}
