// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title  MidstateAtomicSwap
/// @notice ETH side of a cross-chain HTLC atomic swap with the Midstate chain.
///
/// Protocol (maker sells MDS, taker pays ETH; the MAKER generates the secret):
///   1. Maker locks MDS in a Midstate HTLC with hashlock H = BLAKE3(secret),
///      receiver = taker, refund = maker after a LONG timeout.
///   2. Taker verifies that lock on Midstate, then calls lock() here with the
///      same H, beneficiary = maker, and a SHORTER refund delay.
///   3. Maker calls claim(swapId, secret) — revealing `secret` on Base — and
///      receives the ETH. The Claimed event publishes `secret`.
///   4. Taker reads `secret` from the Claimed event and claims the MDS HTLC.
///
///   Safety ordering: the Base refund delay MUST be shorter than the Midstate
///   HTLC timeout, so that after the maker reveals the secret (at the latest
///   just before the Base refund), the taker still has ample time to claim MDS.
contract MidstateAtomicSwap {
    struct Swap {
        address payable beneficiary; // receives ETH on claim (the MDS maker)
        address payable refundTo;    // refunded after timeout (the ETH locker / taker)
        uint256 amount;              // wei locked
        uint64  timeout;             // unix time after which refund() is allowed
        bytes32 hashlock;            // BLAKE3(secret); identical to the Midstate HTLC hashlock
        bool    settled;             // true once claimed or refunded
    }

    mapping(bytes32 => Swap) public swaps;

    event Locked(
        bytes32 indexed swapId,
        address indexed beneficiary,
        address indexed refundTo,
        uint256 amount,
        uint64  timeout,
        bytes32 hashlock
    );
    event Claimed(bytes32 indexed swapId, bytes32 hashlock, bytes32 preimage);
    event Refunded(bytes32 indexed swapId);

    // Minimal reentrancy guard.
    uint256 private _guard = 1;
    modifier nonReentrant() {
        require(_guard == 1, "reentrant");
        _guard = 2;
        _;
        _guard = 1;
    }

    /// @notice Lock ETH, claimable by revealing the BLAKE3 preimage of `hashlock`.
    /// @param hashlock    BLAKE3 hash of the 32-byte secret (same value as the Midstate HTLC).
    /// @param beneficiary address paid on a successful claim (the maker).
    /// @param refundDelay seconds from now after which the locker may reclaim.
    /// @return swapId     unique id for this swap (also emitted in Locked).
    function lock(bytes32 hashlock, address payable beneficiary, uint256 refundDelay)
        external
        payable
        returns (bytes32 swapId)
    {
        require(msg.value > 0, "no value");
        require(beneficiary != address(0), "bad beneficiary");
        require(hashlock != bytes32(0), "bad hashlock");
        require(refundDelay >= 600 && refundDelay <= 7 days, "bad delay");

        uint64 timeout = uint64(block.timestamp + refundDelay);

        // Unique per locker/beneficiary/hashlock/amount/timeout, bound to this
        // chain + contract. Including msg.sender and timeout stops anyone from
        // pre-registering (griefing) the id a counterparty intends to use.
        swapId = keccak256(
            abi.encode(msg.sender, beneficiary, hashlock, msg.value, timeout, address(this), block.chainid)
        );
        require(swaps[swapId].amount == 0, "swap exists");

        swaps[swapId] = Swap({
            beneficiary: beneficiary,
            refundTo: payable(msg.sender),
            amount: msg.value,
            timeout: timeout,
            hashlock: hashlock,
            settled: false
        });

        emit Locked(swapId, beneficiary, msg.sender, msg.value, timeout, hashlock);
    }

    /// @notice Claim the ETH by revealing the 32-byte secret. Pays the beneficiary
    ///         and publishes the secret so the counterparty can claim on Midstate.
    /// @dev    Callable by anyone — knowledge of the secret is the authorization,
    ///         and funds always go to the fixed beneficiary.
    function claim(bytes32 swapId, bytes32 preimage) external nonReentrant {
        Swap storage s = swaps[swapId];
        require(s.amount > 0, "not found");
        require(!s.settled, "settled");
        require(blake3_256(preimage) == s.hashlock, "bad preimage");

        s.settled = true;
        uint256 amt = s.amount;
        address payable to = s.beneficiary;

        emit Claimed(swapId, s.hashlock, preimage);

        (bool ok, ) = to.call{value: amt}("");
        require(ok, "pay failed");
    }

    /// @notice Reclaim the ETH after the timeout if it was never claimed.
    function refund(bytes32 swapId) external nonReentrant {
        Swap storage s = swaps[swapId];
        require(s.amount > 0, "not found");
        require(!s.settled, "settled");
        require(block.timestamp >= s.timeout, "too early");

        s.settled = true;
        uint256 amt = s.amount;
        address payable to = s.refundTo;

        emit Refunded(swapId);

        (bool ok, ) = to.call{value: amt}("");
        require(ok, "refund failed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  BLAKE3 (single 32-byte block, root) — matches Midstate core::types::hash
    //  Validated byte-for-byte against the reference BLAKE3 implementation.
    // ─────────────────────────────────────────────────────────────────────────

    uint256 private constant M = 0xFFFFFFFF;

    function _rotr(uint256 x, uint256 n) private pure returns (uint256) {
        return ((x >> n) | (x << (32 - n))) & M;
    }

    // The BLAKE3 mixing function G, operating in-place on the 16-word state.
    function _g(
        uint256[] memory st,
        uint256 a,
        uint256 b,
        uint256 c,
        uint256 d,
        uint256 mx,
        uint256 my
    ) private pure {
        st[a] = (st[a] + st[b] + mx) & M;
        st[d] = _rotr(st[d] ^ st[a], 16);
        st[c] = (st[c] + st[d]) & M;
        st[b] = _rotr(st[b] ^ st[c], 12);
        st[a] = (st[a] + st[b] + my) & M;
        st[d] = _rotr(st[d] ^ st[a], 8);
        st[c] = (st[c] + st[d]) & M;
        st[b] = _rotr(st[b] ^ st[c], 7);
    }

    /// @notice BLAKE3-256 of a 32-byte input (single block, single chunk, root).
    function blake3_256(bytes32 preimage) public pure returns (bytes32) {
        // 16 little-endian u32 message words: first 8 from the preimage, rest 0.
        uint256[] memory m = new uint256[](16);
        for (uint256 i = 0; i < 8; i++) {
            uint256 base = 4 * i;
            uint256 b0 = uint256(uint8(preimage[base]));
            uint256 b1 = uint256(uint8(preimage[base + 1]));
            uint256 b2 = uint256(uint8(preimage[base + 2]));
            uint256 b3 = uint256(uint8(preimage[base + 3]));
            m[i] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) & M;
        }

        // Initial state. cv = IV; counter = 0; block_len = 32;
        // flags = CHUNK_START | CHUNK_END | ROOT = 1 | 2 | 8 = 11.
        uint256[] memory st = new uint256[](16);
        st[0] = 0x6A09E667; st[1] = 0xBB67AE85; st[2] = 0x3C6EF372; st[3] = 0xA54FF53A;
        st[4] = 0x510E527F; st[5] = 0x9B05688C; st[6] = 0x1F83D9AB; st[7] = 0x5BE0CD19;
        st[8] = 0x6A09E667; st[9] = 0xBB67AE85; st[10] = 0x3C6EF372; st[11] = 0xA54FF53A;
        st[12] = 0;         st[13] = 0;         st[14] = 32;        st[15] = 11;

        // 7 rounds. The (a,b,c,d) lane tuples are identical every round; only the
        // message words rotate, per the precomputed BLAKE3 message schedule.
        // round 0
        _g(st, 0, 4, 8, 12, m[0], m[1]);
        _g(st, 1, 5, 9, 13, m[2], m[3]);
        _g(st, 2, 6, 10, 14, m[4], m[5]);
        _g(st, 3, 7, 11, 15, m[6], m[7]);
        _g(st, 0, 5, 10, 15, m[8], m[9]);
        _g(st, 1, 6, 11, 12, m[10], m[11]);
        _g(st, 2, 7, 8, 13, m[12], m[13]);
        _g(st, 3, 4, 9, 14, m[14], m[15]);
        // round 1  sched [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
        _g(st, 0, 4, 8, 12, m[2], m[6]);
        _g(st, 1, 5, 9, 13, m[3], m[10]);
        _g(st, 2, 6, 10, 14, m[7], m[0]);
        _g(st, 3, 7, 11, 15, m[4], m[13]);
        _g(st, 0, 5, 10, 15, m[1], m[11]);
        _g(st, 1, 6, 11, 12, m[12], m[5]);
        _g(st, 2, 7, 8, 13, m[9], m[14]);
        _g(st, 3, 4, 9, 14, m[15], m[8]);
        // round 2  sched [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
        _g(st, 0, 4, 8, 12, m[3], m[4]);
        _g(st, 1, 5, 9, 13, m[10], m[12]);
        _g(st, 2, 6, 10, 14, m[13], m[2]);
        _g(st, 3, 7, 11, 15, m[7], m[14]);
        _g(st, 0, 5, 10, 15, m[6], m[5]);
        _g(st, 1, 6, 11, 12, m[9], m[0]);
        _g(st, 2, 7, 8, 13, m[11], m[15]);
        _g(st, 3, 4, 9, 14, m[8], m[1]);
        // round 3  sched [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
        _g(st, 0, 4, 8, 12, m[10], m[7]);
        _g(st, 1, 5, 9, 13, m[12], m[9]);
        _g(st, 2, 6, 10, 14, m[14], m[3]);
        _g(st, 3, 7, 11, 15, m[13], m[15]);
        _g(st, 0, 5, 10, 15, m[4], m[0]);
        _g(st, 1, 6, 11, 12, m[11], m[2]);
        _g(st, 2, 7, 8, 13, m[5], m[8]);
        _g(st, 3, 4, 9, 14, m[1], m[6]);
        // round 4  sched [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
        _g(st, 0, 4, 8, 12, m[12], m[13]);
        _g(st, 1, 5, 9, 13, m[9], m[11]);
        _g(st, 2, 6, 10, 14, m[15], m[10]);
        _g(st, 3, 7, 11, 15, m[14], m[8]);
        _g(st, 0, 5, 10, 15, m[7], m[2]);
        _g(st, 1, 6, 11, 12, m[5], m[3]);
        _g(st, 2, 7, 8, 13, m[0], m[1]);
        _g(st, 3, 4, 9, 14, m[6], m[4]);
        // round 5  sched [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
        _g(st, 0, 4, 8, 12, m[9], m[14]);
        _g(st, 1, 5, 9, 13, m[11], m[5]);
        _g(st, 2, 6, 10, 14, m[8], m[12]);
        _g(st, 3, 7, 11, 15, m[15], m[1]);
        _g(st, 0, 5, 10, 15, m[13], m[3]);
        _g(st, 1, 6, 11, 12, m[0], m[10]);
        _g(st, 2, 7, 8, 13, m[2], m[6]);
        _g(st, 3, 4, 9, 14, m[4], m[7]);
        // round 6  sched [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
        _g(st, 0, 4, 8, 12, m[11], m[15]);
        _g(st, 1, 5, 9, 13, m[5], m[0]);
        _g(st, 2, 6, 10, 14, m[1], m[9]);
        _g(st, 3, 7, 11, 15, m[8], m[6]);
        _g(st, 0, 5, 10, 15, m[14], m[10]);
        _g(st, 1, 6, 11, 12, m[2], m[12]);
        _g(st, 2, 7, 8, 13, m[3], m[4]);
        _g(st, 3, 4, 9, 14, m[7], m[13]);

        // 32-byte output: out_i = st[i] ^ st[i+8], serialized little-endian,
        // first byte first (so out[0] is the most-significant byte of the result).
        bytes memory outb = new bytes(32);
        for (uint256 i = 0; i < 8; i++) {
            uint256 w = (st[i] ^ st[i + 8]) & M;
            outb[4 * i]     = bytes1(uint8(w & 0xff));
            outb[4 * i + 1] = bytes1(uint8((w >> 8) & 0xff));
            outb[4 * i + 2] = bytes1(uint8((w >> 16) & 0xff));
            outb[4 * i + 3] = bytes1(uint8((w >> 24) & 0xff));
        }
        bytes32 r;
        assembly {
            r := mload(add(outb, 0x20))
        }
        return r;
    }
}
