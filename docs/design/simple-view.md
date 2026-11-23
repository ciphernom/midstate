# The simple view (desktop app)

Status: built.

New wallets open to seven pages — Home, Send, Receive, Activity, Messages,
Names, Settings — in plain words. Everything else is behind Settings → **Show
all features**, which brings back the full app exactly as it was, every tab
and page unchanged; its Settings has **Use the simple view** to come back. The
choice is kept per profile (`<profile>/view`). The first time this version
opens, a profile that already has a wallet keeps the full app; a new one
starts simple.

## Pages

- **Home**: the balance, Send and Receive, anything under way (catching up
  with the network, finding a restored wallet's coins, sends in progress) and
  the last five payments. With nothing in the wallet, Receive is the main
  button.
- **Send**: to (a name or an address, checked as it is typed), amount, then a
  look at what will be sent before **Send now**, then the send's progress in
  plain steps. A stuck send offers Try again or Cancel. No privacy delay, no
  asking a peer for an address: those are in the full Send page.
- **Receive**: a QR code and a reusable address, made automatically if the
  wallet has none with signatures to spare (a one-time address is never shown
  here: paying one twice loses money), and the wallet's name to share instead.
- **Activity**: every payment in and out by day. Tidying coins together is
  left out; it moves nothing anywhere.
- **Names**: the wallet's names with automatic renewal, and getting a new one
  (or buying one for sale). Selling, renting, records and backing are in the
  full Names page.
- **Settings**: updates, appearance, checking the written recovery phrase,
  locking, and the way to all features.

MDW appears beside MDS (on Home, Send and Receive) only once the wallet's MDW
account is turned on.

## One unit

Every MDS amount is shown and typed in gMDS (2^30 base units, the block reward
the chain started with), which is what the dashboard and the names market
already show. The full Send page defaulted to base units while the dashboard
showed gMDS; the simple view has only the one. Typed amounts round to the
nearest base unit, so 0.1 means 0.1. A comma is a decimal point (2,5) unless
three digits follow it, when 1,234 is refused rather than guessed.

## Also changed for everyone

- Onboarding (create, restore, unlock): shorter words, large buttons. The
  steps and checks are the same, including the three-word check of the written
  phrase, and the phrase screen still says it is shown only once.
- Settings → Updates (see auto-update.md), in both views.
