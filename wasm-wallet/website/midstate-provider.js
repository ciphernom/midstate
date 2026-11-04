/**
 * midstate-provider.js — "MidstateConnect" dApp provider
 * =======================================================
 *
 * The web-native equivalent of `window.ethereum`. A dApp (e.g. the Midstate IDE,
 * or any third-party DEX / marketplace) includes this file and talks to the
 * user's Web Wallet over a `window.postMessage` bridge — no browser extension.
 *
 *   <script src="midstate-provider.js"></script>
 *   <script>
 *     Midstate.configure({ walletUrl: new URL("wallet/", location.href).href });
 *     const { address } = await Midstate.connect();
 *     const { txid }    = await Midstate.fundContract({ contractAddress, amount: 100_000 });
 *     const { txid }    = await Midstate.spendContract({ bytecode, inputs, outputs },
 *                                                       { onProgress: m => console.log(m) });
 *   </script>
 *
 * SECURITY MODEL (read this before changing anything)
 * ---------------------------------------------------
 *  - postMessage does NOT enforce origins by default. We do it manually:
 *      * Every outbound message that carries params uses an EXACT targetOrigin
 *        (the wallet origin) — never "*".
 *      * Every inbound message is rejected unless BOTH event.origin === walletOrigin
 *        AND event.source === the popup we opened.
 *      * A per-request random `id` correlates responses and defeats replay / crosstalk.
 *  - The dApp never sees the seed, password, or raw UTXOs. It only ever receives a
 *    public address (connect) or a transaction id / progress strings.
 *  - The wallet popup is responsible for showing the user WHAT they are approving
 *    (amount, destination, bytecode hash, requesting origin) and requiring a click.
 *    This library cannot and must not auto-approve anything.
 *
 * POPUP / USER-GESTURE RULE
 * -------------------------
 *  window.open() is blocked unless called synchronously inside a user gesture
 *  (a click handler). So every public method here opens the popup FIRST, then does
 *  async negotiation. Call connect()/fundContract()/spendContract() directly from
 *  an onclick — not after an await.
 */
(function (global) {
  "use strict";

  var PROTOCOL = "midstate-bridge";
  var VERSION = 1;

  var config = {
    // Where the wallet is served. Defaults to "<this origin>/wallet/", which is
    // correct when the dApp and wallet share an origin (the Midstate convention).
    // Override via Midstate.configure({ walletUrl }) for a cross-origin wallet.
    walletUrl: (typeof window !== "undefined" && window.location)
      ? new URL("wallet/", window.location.href).href
      : "/wallet/",
    // Exact origin we will accept messages from. Derived from walletUrl if left null.
    walletOrigin: null,
    // Popup geometry.
    popupWidth: 420,
    popupHeight: 640,
    // How long to wait for the popup to say "ready" before giving up.
    handshakeTimeoutMs: 60_000,
  };

  function originOf(url) {
    try { return new URL(url).origin; } catch (_) { return null; }
  }

  function resolvedWalletOrigin() {
    return config.walletOrigin || originOf(config.walletUrl);
  }

  function randomId() {
    if (global.crypto && global.crypto.getRandomValues) {
      var a = new Uint32Array(4);
      global.crypto.getRandomValues(a);
      return Array.from(a, function (n) { return n.toString(16).padStart(8, "0"); }).join("");
    }
    return String(Date.now()) + Math.random().toString(16).slice(2);
  }

  function openCentredPopup(url) {
    var w = config.popupWidth, h = config.popupHeight;
    // dualScreen* fall back across browsers; clamp so the popup is visible.
    var dualLeft = global.screenLeft != null ? global.screenLeft : global.screenX;
    var dualTop = global.screenTop != null ? global.screenTop : global.screenY;
    var width = global.innerWidth || document.documentElement.clientWidth || screen.width;
    var height = global.innerHeight || document.documentElement.clientHeight || screen.height;
    var left = Math.max(0, dualLeft + (width - w) / 2);
    var top = Math.max(0, dualTop + (height - h) / 2);
    var features =
      "popup=yes,width=" + w + ",height=" + h + ",left=" + Math.round(left) +
      ",top=" + Math.round(top) + ",resizable=yes,scrollbars=yes";
    // Named target so repeated calls reuse one window instead of spawning many.
    return global.open(url, "MidstateWallet", features);
  }

  /**
   * Core RPC. Opens the wallet popup (synchronously — call from a click), waits
   * for its "ready" handshake, sends the request, and resolves on "result".
   *
   * @param {string} action   "connect" | "fund" | "spend"
   * @param {object} params   action-specific payload (see public methods)
   * @param {object} [opts]
   * @param {(msg:string)=>void} [opts.onProgress]  streamed status strings
   * @param {number} [opts.resultTimeoutMs]  hard cap on total time (default: none —
   *                 spend can legitimately take minutes while waiting for a block;
   *                 we instead detect a user-closed popup and reset the handshake
   *                 timer whenever progress arrives).
   * @returns {Promise<object>}
   */
  function invoke(action, params, opts) {
    opts = opts || {};
    var walletOrigin = resolvedWalletOrigin();
    if (!walletOrigin) {
      return Promise.reject(new Error("Midstate: walletUrl/walletOrigin is not configured."));
    }

    // 1) Open the popup NOW, while we are still inside the user's click.
    var url = config.walletUrl +
      (config.walletUrl.indexOf("?") === -1 ? "?" : "&") +
      "dapp=1&action=" + encodeURIComponent(action);
    var popup = openCentredPopup(url);
    if (!popup) {
      return Promise.reject(new Error(
        "Midstate: the wallet popup was blocked. Trigger connect/fund/spend directly " +
        "from a user click (not after an await)."
      ));
    }

    return new Promise(function (resolve, reject) {
      var id = randomId();
      var settled = false;
      var sentRequest = false;
      var closedPoll = null;
      var handshakeTimer = null;

      function cleanup() {
        if (settled) return;
        settled = true;
        global.removeEventListener("message", onMessage);
        if (closedPoll) clearInterval(closedPoll);
        if (handshakeTimer) clearTimeout(handshakeTimer);
      }

      function finish(err, value) {
        if (settled) return;
        cleanup();
        try { if (popup && !popup.closed) popup.close(); } catch (_) {}
        if (err) reject(err); else resolve(value);
      }

      function armHandshakeTimer() {
        if (handshakeTimer) clearTimeout(handshakeTimer);
        handshakeTimer = setTimeout(function () {
          finish(new Error("Midstate: wallet did not respond (handshake/idle timeout)."));
        }, config.handshakeTimeoutMs);
      }

      function send(msg) {
        // EXACT targetOrigin — never "*". This is the whole point.
        popup.postMessage(Object.assign({ __mid: PROTOCOL, v: VERSION, id: id }, msg), walletOrigin);
      }

      function onMessage(event) {
        // Hard origin + source gate. Reject everything that isn't our popup.
        if (event.origin !== walletOrigin) return;
        if (event.source !== popup) return;
        var d = event.data;
        if (!d || d.__mid !== PROTOCOL || d.v !== VERSION) return;

        // The wallet's "ready" ping carries no id (it doesn't know ours yet), so
        // handle it BEFORE the id gate. We reply with our id-bearing request; all
        // subsequent frames are correlated by id below.
        if (d.type === "ready") {
          if (!sentRequest) {
            sentRequest = true;
            send({ type: "request", action: action, params: params || {} });
          }
          armHandshakeTimer(); // reset idle timer now that we have contact
          return;
        }

        if (d.id !== id) return; // not our request

        switch (d.type) {
          case "progress":
            armHandshakeTimer(); // progress means it's alive; keep waiting
            if (typeof opts.onProgress === "function") {
              try { opts.onProgress(String(d.message == null ? "" : d.message)); } catch (_) {}
            }
            break;

          case "result":
            finish(null, d.data || {});
            break;

          case "error":
            finish(new Error(d.message || "Midstate: wallet returned an error."));
            break;

          default:
            // ignore unknown frames
            break;
        }
      }

      global.addEventListener("message", onMessage);

      // The popup may not be listening yet, and it must learn OUR id. We can't post
      // until it tells us it's ready (origin/source are only trustworthy on receive),
      // so we wait for its "ready". But to hand it our id before it can send "ready"
      // with the right correlation, the popup reads `id` from the request we send
      // *after* its first ready ping. The handshake timer guards a dead popup.
      armHandshakeTimer();

      // Detect the user closing the popup manually → reject cleanly.
      closedPoll = setInterval(function () {
        if (popup.closed) finish(new Error("Midstate: the user closed the wallet."));
      }, 400);

      // Optional hard overall cap.
      if (opts.resultTimeoutMs) {
        setTimeout(function () {
          finish(new Error("Midstate: request timed out."));
        }, opts.resultTimeoutMs);
      }
    });
  }

  var Midstate = {
    /**
     * Override defaults. Call once before connect().
     * @param {{walletUrl?:string, walletOrigin?:string, popupWidth?:number,
     *          popupHeight?:number, handshakeTimeoutMs?:number}} opts
     */
    configure: function (opts) {
      Object.assign(config, opts || {});
      if (opts && opts.walletUrl && !opts.walletOrigin) config.walletOrigin = null; // re-derive
      return this;
    },

    /** The wallet origin we will trust (for display / debugging). */
    walletOrigin: function () { return resolvedWalletOrigin(); },

    /**
     * Request the user's public MSS address.
     *
     * With `{ siteKey: true }` the wallet also makes (once) a signing key for
     * this site's origin: an MSS key derived from the user's recovery phrase,
     * used only to sign transactions the wallet builds for this site. A
     * contract that names an owner per position (the launchpad's ledger) uses
     * its public key. The first connect takes about 20 seconds; progress
     * arrives through `onProgress`.
     *
     * @param {{siteKey?:boolean, onProgress?:Function, resultTimeoutMs?:number}} [opts]
     * @returns {Promise<{address:string, sitePk?:string, siteKeyLeft?:number,
     *   siteSigBytes?:number, siteKeyGeneration?:number,
     *   sitePks?:Array<{pk:string,left:number,sigBytes:number}>}>}
     *   `sitePks` lists every key this site has had, newest last, so positions
     *   held by an older one can be moved while it still has signatures.
     */
    connect: function (opts) {
      opts = opts || {};
      return invoke("connect", { siteKey: !!opts.siteKey },
        Object.assign({ resultTimeoutMs: opts.siteKey ? 10 * 60_000 : 120_000 }, opts));
    },

    /**
     * Send MDS to a contract's P2SH address (optionally seeding its state).
     * Funding is a standard send under the hood — works today via performSend.
     *
     * @param {object} p
     * @param {string} p.contractAddress  64-char hex P2SH address
     * @param {number|string} p.amount    value in sats
     * @param {string} [p.state]          64-char hex initial state → emits a second
     *                                    `addr:0:state` output (requires the wallet's
     *                                    fund handler to support a state output)
     * @returns {Promise<{txid:string}>}
     */
    fundContract: function (p, opts) {
      // Several payments in one transaction, with optional coin salts and up to
      // 80-byte public data outputs:
      //   { fundings: [{ address, amount, salt? }], burns: [hex], summary? }
      // A salt can only be chosen for an amount that is one power of two.
      if (p && Array.isArray(p.fundings)) {
        if (!p.fundings.length) return Promise.reject(new Error("Midstate.fundContract: fundings is empty."));
        return invoke("fund", {
          fundings: p.fundings.map(function (f) {
            return { address: f.address, amount: String(f.amount), salt: f.salt || null };
          }),
          burns: p.burns || [],
          summary: p.summary || null,
        }, Object.assign({ resultTimeoutMs: 5 * 60_000 }, opts));
      }
      if (!p || !p.contractAddress) {
        return Promise.reject(new Error("Midstate.fundContract: contractAddress is required."));
      }
      return invoke("fund", {
        contractAddress: p.contractAddress,
        amount: String(p.amount),
        state: p.state || null,
      }, Object.assign({ resultTimeoutMs: 5 * 60_000 }, opts));
    },

    /**
     * Execute (spend) a contract with an arbitrary witness stack. Backed by the
     * wallet's `prepare_script_spend` + `build_script_reveal`. Params are forwarded
     * verbatim; the wallet resolves the on-chain contract coins itself.
     *
     * Typical (IDE) shape:
     *   { bytecode, contractAddress, stateWitness, inputState, valueWitness,
     *     outputs: [{ address, value, state? }] }
     * Advanced dApps may pass explicit { inputs: [{ coinId, witness, inputState }] }.
     *
     * A signature over THIS transaction's commitment comes from the site key
     * (see connect({siteKey: true})): pass `siteSig: { item, pk }` and leave,
     * at witness position `item` of every contract input, a placeholder of
     * `siteSigBytes` zero bytes. The wallet builds the transaction, signs its
     * commitment with that key and puts the signature in. Without `siteSig`,
     * preimage / timelock / routing / state-covenant paths work as before.
     *
     * Explicit inputs: `inputs: [{ coinId, value, salt, witness }]`, the witness
     * comma-separated hex, pushed left to right. Outputs:
     * `[{ out_type: "standard"|"confidential", address, value, state?, salt? }]`;
     * whatever the outputs need beyond the inputs is paid from the wallet, and
     * anything left over comes back to it. `summary` is shown to the user.
     *
     * @param {object} p   must include `bytecode`
     * @returns {Promise<{txid:string}>}
     */
    spendContract: function (p, opts) {
      if (!p || !p.bytecode) {
        return Promise.reject(new Error("Midstate.spendContract: bytecode is required."));
      }
      // The WALLET defines the spend params it understands (it resolves the
      // on-chain contract coins itself), so we forward `p` verbatim rather than
      // locking dApps into one witness model. Typical shape from the IDE:
      //   { bytecode, contractAddress, stateWitness, inputState, valueWitness,
      //     outputs: [{ address, value, state? }] }
      // Advanced dApps may instead pass explicit
      //   { inputs: [{ coinId, witness, inputState }] }.
      return invoke("spend", p, opts); // no default hard timeout: spend waits for a block
    },

    PROTOCOL: PROTOCOL,
    VERSION: VERSION,
  };

  // Expose as a global for classic <script> dApps...
  global.Midstate = Midstate;
  // ...and support module consumers without breaking the global.
  if (typeof module !== "undefined" && module.exports) module.exports = Midstate;
})(typeof window !== "undefined" ? window : this);
