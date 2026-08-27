const { compile, execute, H } = await import('/tmp/mstest/stage.mjs');
const { pumpContractSourceV6 } = await import('/home/claude/showcase/pump-v6.mjs');
const P = await import('midstate-sdk/pump');
const L = await import('midstate-sdk/launcher');
const w = await import('midstate-sdk/pkg/wasm_wallet.js');

const DEPTH = 4, THRESH = 10n, POOL = 50n, AMM = 'ee'.repeat(32);
const ASSET = L.deriveAssetId('MYCOIN', 'ab');
const c = compile(pumpContractSourceV6({ depth:DEPTH, threshold:THRESH, ammAddr:AMM, assetId:ASSET, poolTokens:POOL }), { height:H });
const ADDR = w.blake3_hash_hex(c.bytecode);
const ZERO = '00'.repeat(24);

console.log(`pump v6 + graduation — ${c.sizeBytes} B / 1024, ${c.sigops} sigops`);
console.log(`threshold ${THRESH} tokens, depth ${DEPTH}, asset ${ASSET.slice(0,12)}...\n`);

const ledger = new P.PumpLedger(DEPTH);
let supply = 0n, graduated = false;
const K = { alice:'aa'.repeat(32), bob:'bb'.repeat(32) };
const res = (s) => (s*s - s)/2n;
const flat = (i) => ledger.proof(i).map(([x,d]) => x+d).join('');
const st = (s,r) => P.encodePumpState(s, r);
const le = P.minimalLE;
const line = (n,ok,err) => console.log(`  ${(ok?'ACCEPT':'REJECT').padEnd(7)} ${n.padEnd(36)} ${err||''}`);

const exec = (wit, outs, sumIn) => execute(c.asm, { height:H, witness:wit,
  inputState: st(supply, ledger.root), thisAddress: ADDR,
  sumInputValue: Number(sumIn ?? (graduated ? 0n : res(supply))), outputs: outs });

function trade(user, slot, amt, side) {
  const pk = K[user], own = P.ownerFromPubkey(pk);
  const oldBal = ledger.balances[slot];
  const ownerOld = oldBal === 0n ? ZERO : ledger.owners[slot];
  const proof = flat(slot), n = BigInt(amt);
  const newBal = side==='buy' ? oldBal+n : oldBal-n;
  const ns = side==='buy' ? supply+n : supply-n;
  const bak = ledger.balances.slice(), ow = ledger.owners.slice();
  ledger.owners[slot] = own; ledger.set(slot, newBal);
  const newRoot = ledger.root;
  ledger.balances = bak; ledger.owners = ow; ledger._rebuild();
  const r = exec([ownerOld, own, pk, le(oldBal), le(n), pk, proof, side==='buy'?'00':'01'],
    [{address:ADDR,value:0,state:st(ns,newRoot)}, {address:ADDR,value:Number(res(ns))}]);
  if (r.ok) { ledger.owners[slot] = own; ledger.set(slot, newBal); supply = ns; }
  return r;
}

console.log('phase 1 — curve live');
line('alice buys 5', trade('alice',0,5,'buy').ok, '');
line('bob buys 5',   trade('bob',1,5,'buy').ok, '');
console.log(`          supply ${supply}/${THRESH}, reserve ${res(supply)}`);
let r = trade('alice',0,1,'buy'); line('buy past the threshold', r.ok, r.error);

console.log('\nphase 2 — graduation (permissionless)');
const ammState = L.encodeState(L.LAYOUTS.AMM, { reserveX: res(supply), reserveY: POOL, padding: '' });
const gradOuts = [{address:ADDR,value:0,state:st(supply,ledger.root)},
                  {address:AMM,value:0,state:ammState},
                  {address:AMM,value:Number(res(supply))}];
const gradWit = [ZERO,ZERO,'00'.repeat(32),'','','','00'.repeat(33*DEPTH),'02'];
r = exec(gradWit, gradOuts); line('graduate: reserve -> AMM', r.ok, r.error);
if (r.ok) graduated = true;
r = exec(gradWit, gradOuts, 0n); line('graduate again (replay)', r.ok, r.error);
r = trade('alice',0,1,'sell'); line('sell after graduation', r.ok, r.error);

console.log('\nphase 3 — redemption to a transferable token');
function redeem(user, slot, bad = {}) {
  const pk = K[user], own = P.ownerFromPubkey(pk);
  const bal = ledger.balances[slot], proof = flat(slot);
  const bak = ledger.balances.slice(), ow = ledger.owners.slice();
  ledger.owners[slot] = ZERO; ledger.set(slot, 0n);
  const newRoot = ledger.root;
  ledger.balances = bak; ledger.owners = ow; ledger._rebuild();
  const tok = L.encodeState(L.LAYOUTS.TOKEN, { balance: bad.bal ?? bal, assetID: bad.asset ?? ASSET });
  const rr = exec([own, own, bad.pk ?? pk, le(bal), '', bad.sig ?? pk, proof, '03'],
    [{address:ADDR,value:0,state:st(supply,newRoot)}, {address:'cd'.repeat(32),value:0,state:tok}]);
  if (rr.ok && !bad.keep) { ledger.owners[slot] = ZERO; ledger.set(slot, 0n); }
  return rr;
}
r = redeem('alice',0); line('alice redeems 5 -> token UTXO', r.ok, r.error);
r = redeem('bob',1,{ bal: 99n, keep:true }); line('bob inflates the token balance', r.ok, r.error);
r = redeem('bob',1,{ asset:'ff'.repeat(24), keep:true }); line('bob mints the wrong asset', r.ok, r.error);
r = redeem('bob',1,{ sig:'cc'.repeat(32), keep:true }); line('forged signature', r.ok, r.error);
r = redeem('bob',1); line('bob redeems 5 -> token UTXO', r.ok, r.error);
console.log(`          ledger now empty: ${ledger.balances.every(b=>b===0n)}, supply marker ${supply}`);
