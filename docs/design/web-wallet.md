# The web wallet

Status: built. Sealed messages and names are written and tested against
stand-ins for their WebAssembly in the browser (and natively against the real
crypto and the registry's own covenant); the real WebAssembly build is to be
checked (below).

The browser wallet (`wasm-wallet/`, served at `/wallet/`) is a new page on the
same engine. Eight sections, as midwimble's web wallet has them and in
midstate's black and white: **Home** (balance, Send and Receive, anything under
way, the last five payments), **Activity** (every payment by day, each with its
details), **Send**, **Receive**, **Messages** (the public room, and sealed
messages), **Names**, **Mine** and **Settings**. On a
phone there are no tabs: Home links everywhere and each page has a way back.
The icon is the desktop app's (`icon-192.png` and `icon-512.png` are its
`logo-1024.png`, resized).

The engine is unchanged in what it does: `worker.js` holds the keys and builds,
signs and sends in WebAssembly; `net.js` (moved out of the old page) carries
its requests over WebRTC or the HTTPS gateway; `mining.js` (also moved out)
runs the miners. The page is `index.html` + `app.js`; `ui.js` has its
building blocks, `format.js` its amounts, addresses and times, `names.js`
names and their proofs, `spend.js` "send all", `bridge.js` MidstateConnect,
`messages.js` (run by the worker) sealed messages, `room.js` the public room,
`registrar.js` (run by the worker) the names this wallet holds.
Nothing is loaded from anywhere else, and the page sets a content security
policy.

## As the desktop's simple view

- **One unit.** Every MDS amount is shown and typed in gMDS (2^30 base
  units). A review shows every decimal an amount has, so 4.499999 never reads
  as 4.5.
- **Safe addresses.** Send takes only a checksummed (72-character) address or
  a name, checked as it is typed: a 64-character address is refused with the
  reason, a typo is caught by the checksum, a midwimble address is pointed to
  a midwimble wallet. Receive shows the reusable address in groups of four,
  with a QR code, and warns once it has fewer than 64 signatures left.
- **Pay a name.mds.** The worker fetches the node's proof for the name and
  checks it here before anything is signed (`names.js`, as midwimble does):
  the leaf against the registry root, the root against a registry coin, the
  coin live, the name not lapsing within a day, and for a name that pays an
  MSS key, signatures to spare. The name is looked up once more when Send is
  pressed; if it now pays somewhere else, the review says so and waits.
- **Onboarding.** Create: the 24 words shown once, three of them typed back
  (a wrong one sends you back to the paper), then a password. Restore: the 24
  words (numbering from a pasted list is ignored; the words survive a typo)
  and a password. A wallet made from a mining link starts mining when it
  opens. Unlock, and a backup file or a command-line wallet file, as before.
- **Send.** To, amount (with Max), review with the network fee the worker
  works out without signing, then the send's progress in three plain steps.

- **Wallet care** (Settings): *Merge small coins* consolidates an address's
  coins (up to 5,000, one signature for all of them) onto the main address,
  for the dust mining leaves; it shows how many coins become how many and the
  fee before anything is signed, and Home offers it once 50 coins could be
  merged. *Gather your coins* defragments; *Look through the chain again*
  rescans.

Showing the 24 words asks for the password again. The update notice no longer
appears on everyone's first visit, only when a new version takes over a page
that is open.

## What's left to the classic page

Trades on the OTC market, Q-Bolt channels and mining bonds are not on the new
page. The old page is kept as `classic.html`, unchanged
but for a link back and the tab lock below, so anything open there can still
be finished: the new page links to it (Home, Settings, and the unlock screen)
while this browser holds an unfinished trade, a channel that isn't closed or a
bond still holding coins. Backups made by either page carry the trades; a
backup's trades are added to the ones already in the browser, never replacing
one (a trade's record holds the only copy of the salt that reclaims it).

## One tab at a time

Two pages signing from one wallet could reuse a one-time key. Both pages take
a Web Lock (`midstate-wallet`) before the wallet starts; a second tab says the
wallet is open elsewhere and offers to try again. So does a dApp's popup while
the wallet is open in another tab: close that tab, then try again.

## Changes to the worker

New requests, all answered by id: `RESOLVE_NAME` (→ `NAME_RESOLVED` /
`NAME_ERROR`), `CHECK_ADDRESS` (→ `ADDRESS_CHECKED`), `QUOTE_SEND`
(→ `SEND_QUOTE`: the fee, and for "send all" the amount, from the same coin
choice as the send), `QUOTE_CONSOLIDATE` (→ `CONSOLIDATE_QUOTE`), `MSG`
(sealed messages and the public room: `{ op, … }` → `MSG_ANSWER`; the worker
also posts `MSG_BOOK`, the conversations, `MSG_NOTICE`, and `ROOM`, the
room) and `NAMES` (`{ op, … }` → `NAMES_ANSWER`; the worker posts `NAMES`,
the names it holds and their jobs, and `NAMES_NOTICE`). `REVEAL_SEED` with a password answers `SEED_REFUSED` if
it's wrong; without one (the classic page) it behaves as before. The
dashboard carries `restoring` while a restored phrase's first scan runs.

Two fixes the classic page gets too:

- **Send all** could fail with "Insufficient funds" when the wallet held many
  coins: the fee depends on the number of outputs, which depends on the
  amount, so the fixed point circled, and its fallback (the last fee plus a
  2-unit buffer) then didn't fit. `spend.js` now picks the amounts to try
  from the fee rule (each probe can cost seconds, since every change coin
  needs a one-time key), and falls back to the old loop and then a search.
  Usually the first amount tried fits and leaves no change.
- **Tidying up** (defragment, and consolidating an address) read
  `input_coin_ids` from a context that has never had it, so after the reveal
  was broadcast every tidy-up was reported as failed and the wallet didn't
  rescan. It now reads the first input from the commitment.

## Deploying

`.github/workflows/deploy-wallet.yml` copies the new files, builds the QR
reader with the other bundles (`qr_reader.bundle.js`, jsQR pinned at 1.4.0)
and runs `npm test` (format, names, send all, messages, the room, the
registrar; against the WebAssembly it has just built). `check-site.js` now also fails the deploy for a file a page loads
by `src`/`href` or a web app manifest names as an icon: the manifest had
pointed at two icons that didn't exist. `sw.js` precaches the page's modules.

## Sealed messages

The desktop's sealed messages (`src/sealed.rs`, run by
`crates/walletd/src/service_identity.rs`), in the browser:

- **The same protocol.** A name's wallet is asked for its KEM key over the
  chat bus (`CMD_KEM_REQ`, answered by `CMD_KEM`), and the key is checked
  against the name's `kem` record; one session carries a conversation; a
  stranger's messages carry a stamp of the work the recipient's name asks
  (`postage`, 24 bits where it says none) until the other side writes back;
  requests hold three messages, go after 30 days, and at most 100 are kept;
  verified names can be blocked, anonymous senders dropped; timers delete
  messages from both wallets. The desktop app and this page read and write
  each other's messages.
- **The same key.** The KEM key comes from the master seed exactly as walletd
  derives it, so a recovery phrase restored here reads what was sealed to the
  names it set up in the desktop app.
- **Write as a name.** Told which name (Message settings), the wallet checks
  that its identity key is one of this wallet's MSS keys and its `kem` record
  this wallet's key. Then it signs new conversations as the name (one leaf of
  that key each, chosen and saved as every other signature here is) and hands
  its key to whoever asks the name's identity, at most 8 a block, and not for
  requests older than 30 minutes. Without a name it writes anonymously. A
  name that stops listing this wallet's keys is let go, with a notice.
- **WebAssembly** (`wasm-wallet/src/messages.rs`): the wallet's KEM key, its
  trial decryption, and openings (`WebWallet` methods); sessions, stamps and
  key checks as functions over the JSON `sealed::Session` and
  `sealed::Prepared` serialize to. Stamps and the chat bus's proof of work are
  searched a slice at a time (wasm has no threads), the worker yielding
  between slices; the chat proof of work hashes the chat once and then only
  the nonce's last block, and checks what it finds with
  `chat::verify_chat_pow_v2`.
- **The worker** (`messages.js`, driven by `worker.js`) keeps the conversations
  in the wallet's saved state, encrypted with the rest of it. It reads the bus
  every 15 seconds while Messages is on screen, every minute otherwise, and as
  the node pushes chats over WebRTC. Each chat it sends is mined here and goes
  from a fresh random sender, from a random nonce. A chat the node refuses is
  tried again for an hour, then shown as not sent with *Try again*.
- **Where it differs from the desktop.** Nodes keep the last 100 chats and
  the page reads them only while it is open, so a message that comes and goes
  while it is closed is missed. What strangers pay is the name's `postage`
  record: changed in Names for a name this wallet holds, in the desktop app
  for one it holds. Conversations keep 400 lines each and 300 kB of text in
  all, oldest first, for the browser's storage. A WebAssembly build without
  the sealed functions leaves sealed messages out of the page rather than
  breaking the wallet; the public room stays.
- **Wire and consensus:** nothing new. Commands 69/70 and the
  `EncryptedMessage` attachment are the desktop's; no `Message` or
  `ChatAttachment` variant is added; older nodes never see a sealed chat (the
  node sends it only to peers that advertise sealed chat).

Tests: `cargo test --lib messages` in `wasm-wallet/` (native: the key is the
desktop's, an opening and a conversation as the page drives them, a signed
opening spends one leaf and verifies, the chat proof of work is the node's,
stamp slices); `tests/messages.test.mjs` (wallets talking over a pretend bus:
the handshake, prices, requests, blocking, timers, unsend and cancel, a
reload, a refused chat, writing as a name, storage limits).

## The public room

The chat bus as everyone sees it, pinned at the top of Messages: up to ten
of its 256 fixed words a message (`chat::CHAT_DICTIONARY`, copied into
`room.js`; a test checks the copy against `src/chat.rs`), and addresses.
Wire frames (words starting with 255) and sealed messages ride the same bus
and are left out, as the classic page left them. One read of `/api/chat`
feeds the room and sealed messages both.

- **Saying something.** Type words, or pick them: the words that start with
  what is being typed show under the box, and all 256 are a tap away. A word
  the room doesn't have is named; so is a message that would start with ⚡
  (wallets would read it as a payment-channel frame). The wallet's receive
  address can go with it, and a message can answer another.
- **Who says it.** A random id the wallet keeps (`wState.roomSender`), tied
  to none of its keys or coins. The classic page used the newest MSS key,
  which told the room whose coins were whose. Nicknames are this browser's
  own, kept where the classic page kept them.
- **Proof of work** is the chat's 20 bits, done here a slice at a time
  (`chat_pow_search`), or in one go with a WebAssembly build that predates it.

## Names

A name.mds held by this wallet itself, with no desktop app: registered (or
claimed once lapsed, or a premium name no one has claimed), renewed, and
changed, from **Names**. A name pays this wallet, and sealed messages to it
reach this wallet; the first one taken is the one messages are written as.

- **No node change.** A node's `GET /name/:name/proof` (one leaf and its path
  to the registry's root) is all a spend needs. A change of a name rebuilds
  its own path. A registration also needs an empty slot's path, and takes it
  from the same proof: the largest subtree beside the path that the proof
  shows empty holds nothing, so every sibling below it is the empty
  subtree's hash and the ones above are on the path already. The covenant
  takes any empty slot. Followers fill the lowest free slots for their own
  wallets; these are far from them.
- **Checked before anything is signed.** `names.js` checks the answer (the
  leaf's key, or for a free name that its neighbour's link skips over it;
  the records and terms against the leaf; the proof against the root; the
  root against a registry coin), and `wasm-wallet/src/registry.rs` checks it
  again as it plans the spend: `name_plan` gives the registry input's
  witness, output 0 (the thread's next coin, its salt by MSN-1's convention)
  and the burns. The page builds and signs it as the IDE's contract spends
  are built (`prepare_script_spend`, which now takes `data_burn` outputs, and
  `build_script_reveal`).
- **The wallet's own keys, as the desktop's.** A name is held (its leaf's
  `owner`) by a fresh MSS key from the seed, the next HD index, height 10:
  what walletd's `fresh_key` makes. Any name held by one of the wallet's MSS
  keys is its own, as walletd counts them, so the same 24 words manage the
  same names in either app. Making the key takes about a minute in a
  browser. A key with four signatures left hands the name to a fresh key in
  the same spend. A restored wallet, or one whose names were taken in the
  desktop app, finds them when asked (*Find a name this wallet holds*): its
  own keys first, then the seed's next four MSS indices (walletd's
  `MSS_GAP`).
- **What a name says.** Taken here: payments to the receive key (the newest
  MSS key with signatures to spare), the identity record (the channel
  identity, as the desktop app's), this wallet's sealed-message key, a line
  of text, and what strangers pay if not the standard. A change keeps every
  record this wallet doesn't read.
- **Seen through.** `registrar.js` keeps each name's job with the wallet and
  sees it through, one spend at a time with the wallet's others: read the
  proof, plan, build, reserve the coins and key counters, commit; once the
  commit is mined and the registry coin is still there (and no other change
  waits for a block), sign and reveal; then check whose spend took the
  registry: this one's if its fee coins are gone too. Another wallet's change
  getting in first is ordinary (every change spends the one registry coin):
  the job starts over from a fresh proof at the next block, up to eight
  tries. A job survives a reload: one committed is seen through from where it
  was, re-sending the reveal it made rather than signing again.
- **One-time keys.** What is signed is signed late, so a spend that lost the
  race has usually signed nothing. If a reveal went out and can't land, its
  MSS coins sign again from a new leaf, which is safe; its WOTS coins can't,
  so they are locked as the wallet locks coins a reorganisation returns, and
  the page says so. Fees come from MSS coins when they cover them, which
  keeps that rare.
- **What it costs.** A year of a name burns its tier's rate (64, 8, 1 or ⅛
  gMDS); each change pays a network fee; the page says both before anything
  is signed, and that the burn goes to no one.

Tests: `cargo test --lib registry` in `wasm-wallet/` (native: plans from a
single proof equal the full-tree spends for registrations and changes, a
premium name is claimed as a lapse, bad proofs are refused, and a name built
as the page builds it, registered and then renewed by its key (an HD MSS key
of the seed), runs through the registry's own covenant; another key's
signature is refused);
`tests/registrar.test.mjs` (jobs against a pretend chain: a registration, a
race lost before and after signing, a renewal's signature in its place, an
edit and a handover, a lapsed name claimed, a name taken first, a reload
mid-spend, a reveal refused, retries, a name found among the seed's keys,
a name lost);
`tests/room.test.mjs` (the dictionary, what is public, typing words).
