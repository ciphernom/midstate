// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// The ERC-20 calls this contract makes. USDC returns a bool from transfer
/// and transferFrom; tokens that return nothing are taken too (see _call).
interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

/// EIP-2612: an allowance signed off-chain, so paying in takes one transaction.
interface IERC20Permit {
    function permit(address owner, address spender, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external;
}

/// @title  MidstateTokenSwap
/// @notice The token side of cross-chain HTLC atomic swaps with the Midstate
///         chain: MidstateAtomicSwap's protocol, rule for rule, for one ERC-20
///         (USDC on Base) fixed when the contract is deployed.
///
///         Everything MidstateAtomicSwap says of its two order shapes holds
///         here, with token units in place of wei:
///           • lock/claim/refund — the taker's token leg of a SELL-MDS order
///             (the resting order is the MDS lock on Midstate);
///           • createBid/reserveBid/claimBid/cancelBid — resting BUY-MDS
///             orders, the tokens escrowed here up front.
///         The events, and the `swaps` and `bids` getters, are the same as
///         MidstateAtomicSwap's, so one reader serves both contracts; a
///         wallet tells them apart by the address that emitted them.
///
///         Differences, all forced by tokens:
///           • Tokens are pulled with transferFrom, so the payer approves this
///             contract first, or signs an EIP-2612 permit and calls the
///             `...WithPermit` form, which does both in one transaction.
///           • A bid's fill bond is in the token too, so it is weighed against
///             the escrow like for like; a seller with no tokens fills bids
///             that ask no bond.
///           • A token that delivers less than it was asked to move (a fee on
///             transfer) is refused: what is locked is exactly what was asked.
///
///         Not for every token: one that rebases, or that a holder's balance
///         can shrink inside, breaks the escrow. Deploy it for USDC, or for a
///         token that behaves as USDC does.
contract MidstateTokenSwap {
    /// The token every amount here is counted in.
    IERC20 public immutable token;

    constructor(IERC20 token_) {
        require(address(token_).code.length > 0, "not a token");
        token = token_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  SELL-MDS ORDERS: the taker locks tokens against the maker's hash
    // ─────────────────────────────────────────────────────────────────────────

    struct Swap {
        address payable beneficiary; // receives the tokens on claim (the MDS maker)
        address payable refundTo;    // refunded after timeout (the token locker / taker)
        uint256 amount;              // token units locked
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

    uint256 private _guard = 1;
    modifier nonReentrant() {
        require(_guard == 1, "reentrant");
        _guard = 2;
        _;
        _guard = 1;
    }

    /// @notice Lock `amount` tokens (approved to this contract beforehand),
    ///         claimable by revealing the BLAKE3 preimage of `hashlock`.
    /// @param hashlock    BLAKE3 hash of the 32-byte secret (the Midstate HTLC's).
    /// @param beneficiary address paid on a successful claim (the maker).
    /// @param amount      token units to lock.
    /// @param refundDelay seconds from now after which the locker may reclaim.
    function lock(bytes32 hashlock, address payable beneficiary, uint256 amount, uint256 refundDelay)
        external
        nonReentrant
        returns (bytes32 swapId)
    {
        return _lock(hashlock, beneficiary, amount, refundDelay);
    }

    /// @notice `lock`, approving the tokens with an EIP-2612 permit signed by
    ///         the caller for exactly `amount`, in the same transaction.
    function lockWithPermit(
        bytes32 hashlock,
        address payable beneficiary,
        uint256 amount,
        uint256 refundDelay,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external nonReentrant returns (bytes32 swapId) {
        _permit(amount, deadline, v, r, s);
        return _lock(hashlock, beneficiary, amount, refundDelay);
    }

    function _lock(bytes32 hashlock, address payable beneficiary, uint256 amount, uint256 refundDelay)
        private
        returns (bytes32 swapId)
    {
        require(amount > 0, "no value");
        require(beneficiary != address(0), "bad beneficiary");
        require(hashlock != bytes32(0), "bad hashlock");
        require(refundDelay >= 600 && refundDelay <= 7 days, "bad delay");

        uint64 timeout = uint64(block.timestamp + refundDelay);

        // Unique per locker/beneficiary/hashlock/amount/timeout, bound to this
        // chain + contract, as MidstateAtomicSwap's: nobody can take the id a
        // counterparty means to use.
        swapId = keccak256(
            abi.encode(msg.sender, beneficiary, hashlock, amount, timeout, address(this), block.chainid)
        );
        require(swaps[swapId].amount == 0, "swap exists");

        swaps[swapId] = Swap({
            beneficiary: beneficiary,
            refundTo: payable(msg.sender),
            amount: amount,
            timeout: timeout,
            hashlock: hashlock,
            settled: false
        });

        _pull(amount);
        emit Locked(swapId, beneficiary, msg.sender, amount, timeout, hashlock);
    }

    /// @notice Claim the tokens by revealing the 32-byte secret. Pays the
    ///         beneficiary and publishes the secret, so the counterparty can
    ///         claim on Midstate.
    /// @dev    Callable by anyone: knowing the secret is the authorization,
    ///         and the tokens always go to the fixed beneficiary.
    function claim(bytes32 swapId, bytes32 preimage) external nonReentrant {
        Swap storage s = swaps[swapId];
        require(s.amount > 0, "not found");
        require(!s.settled, "settled");
        require(blake3_256(preimage) == s.hashlock, "bad preimage");

        s.settled = true;
        emit Claimed(swapId, s.hashlock, preimage);
        _push(s.beneficiary, s.amount);
    }

    /// @notice Take the tokens back after the timeout, if never claimed.
    function refund(bytes32 swapId) external nonReentrant {
        Swap storage s = swaps[swapId];
        require(s.amount > 0, "not found");
        require(!s.settled, "settled");
        require(block.timestamp >= s.timeout, "too early");

        s.settled = true;
        emit Refunded(swapId);
        _push(s.refundTo, s.amount);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  RESTING BIDS: buy-MDS limit orders, tokens escrowed up front
    //
    //  The protocol, the safety ordering and the one-reservation rule are
    //  MidstateAtomicSwap's (see its RESTING BIDS section): the bid's maker
    //  holds the secret and reveals it on Midstate by claiming the seller's
    //  covenant; the seller then collects the escrow here with it, before the
    //  reservation's deadline. A bid is reserved once, ever, and a replacement
    //  takes a fresh secret (_hashlockUsed).
    //
    //  The fill bond is in the token: the seller stakes it when reserving and
    //  gets it back with the escrow, or forfeits it to the maker if the
    //  reservation lapses unclaimed.
    // ─────────────────────────────────────────────────────────────────────────

    struct Bid {
        address payable maker;   // token escrower (MDS buyer); refunds + forfeited bonds go here
        address payable filler;  // seller holding the (single) reservation; 0 while open
        uint256 amount;          // token units escrowed for the seller
        uint256 fillBond;        // token units a seller must stake to reserve (may be 0)
        uint64  mdsAmount;       // MDS units the maker expects locked on Midstate
        uint64  expiry;          // unix time after which the bid can no longer be reserved
        uint64  fillDeadline;    // unix time the active reservation (and claim window) ends
        bytes32 hashlock;        // BLAKE3(secret); the secret is generated and held by the MAKER
        bytes32 makerMdsAddr;    // Midstate address the seller's covenant must pay
        bool    settled;         // true once claimed or cancelled
    }

    mapping(bytes32 => Bid) public bids;
    uint256 private _bidNonce;
    mapping(bytes32 => bool) private _hashlockUsed;

    event BidCreated(
        bytes32 indexed bidId,
        address indexed maker,
        bytes32 hashlock,
        uint256 amount,
        uint256 fillBond,
        uint64  mdsAmount,
        bytes32 makerMdsAddr,
        uint64  expiry
    );
    event BidReserved(bytes32 indexed bidId, address indexed filler, uint64 fillDeadline);
    event BidClaimed(bytes32 indexed bidId, bytes32 hashlock, bytes32 preimage);
    event BidCancelled(bytes32 indexed bidId);

    /// @notice Escrow `amount` tokens (approved beforehand) as a resting buy
    ///         order for `mdsAmount` MDS.
    /// @param hashlock     BLAKE3 hash of a fresh 32-byte secret held by the caller.
    /// @param amount       token units escrowed for the filling seller.
    /// @param mdsAmount    MDS units the filling seller must lock on Midstate.
    /// @param makerMdsAddr Midstate address the seller's covenant HTLC must pay.
    /// @param ttl          seconds this bid stays reservable (1 hour – 90 days).
    /// @param fillBond     token units a seller must stake in reserveBid (0 for none).
    function createBid(
        bytes32 hashlock,
        uint256 amount,
        uint64  mdsAmount,
        bytes32 makerMdsAddr,
        uint256 ttl,
        uint256 fillBond
    ) external nonReentrant returns (bytes32 bidId) {
        return _createBid(hashlock, amount, mdsAmount, makerMdsAddr, ttl, fillBond);
    }

    /// @notice `createBid`, approving the escrow with an EIP-2612 permit for
    ///         exactly `amount`, in the same transaction.
    function createBidWithPermit(
        bytes32 hashlock,
        uint256 amount,
        uint64  mdsAmount,
        bytes32 makerMdsAddr,
        uint256 ttl,
        uint256 fillBond,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external nonReentrant returns (bytes32 bidId) {
        _permit(amount, deadline, v, r, s);
        return _createBid(hashlock, amount, mdsAmount, makerMdsAddr, ttl, fillBond);
    }

    function _createBid(
        bytes32 hashlock,
        uint256 amount,
        uint64  mdsAmount,
        bytes32 makerMdsAddr,
        uint256 ttl,
        uint256 fillBond
    ) private returns (bytes32 bidId) {
        require(amount > 0, "no value");
        require(mdsAmount > 0, "no mds amount");
        require(hashlock != bytes32(0), "bad hashlock");
        require(!_hashlockUsed[hashlock], "hashlock reused");
        require(makerMdsAddr != bytes32(0), "bad mds addr");
        require(ttl >= 1 hours && ttl <= 90 days, "bad ttl");

        _hashlockUsed[hashlock] = true;
        uint64 expiry = uint64(block.timestamp + ttl);

        bidId = keccak256(
            abi.encode(msg.sender, hashlock, amount, _bidNonce++, address(this), block.chainid)
        );
        require(bids[bidId].amount == 0, "bid exists");

        bids[bidId] = Bid({
            maker: payable(msg.sender),
            filler: payable(address(0)),
            amount: amount,
            fillBond: fillBond,
            mdsAmount: mdsAmount,
            expiry: expiry,
            fillDeadline: 0,
            hashlock: hashlock,
            makerMdsAddr: makerMdsAddr,
            settled: false
        });

        _pull(amount);
        emit BidCreated(bidId, msg.sender, hashlock, amount, fillBond, mdsAmount, makerMdsAddr, expiry);
    }

    /// @notice Reserve a bid BEFORE locking MDS on Midstate, staking its fill
    ///         bond (approved beforehand, if it asks one). Exclusive until
    ///         fillDeadline; each bid can be reserved exactly once.
    /// @param fillWindow seconds of exclusivity (2 hours – 3 days). Pick the
    ///        Midstate covenant timeout to end WELL BEFORE now + fillWindow.
    function reserveBid(bytes32 bidId, uint256 fillWindow) external nonReentrant {
        _reserveBid(bidId, fillWindow);
    }

    /// @notice `reserveBid`, approving the bond with an EIP-2612 permit for
    ///         exactly the bid's fill bond, in the same transaction.
    function reserveBidWithPermit(bytes32 bidId, uint256 fillWindow, uint256 deadline, uint8 v, bytes32 r, bytes32 s)
        external
        nonReentrant
    {
        _permit(bids[bidId].fillBond, deadline, v, r, s);
        _reserveBid(bidId, fillWindow);
    }

    function _reserveBid(bytes32 bidId, uint256 fillWindow) private {
        Bid storage b = bids[bidId];
        require(b.amount > 0, "not found");
        require(!b.settled, "settled");
        require(b.filler == address(0), "already reserved");
        require(block.timestamp < b.expiry, "expired");
        require(fillWindow >= 2 hours && fillWindow <= 3 days, "bad window");

        b.filler = payable(msg.sender);
        b.fillDeadline = uint64(block.timestamp + fillWindow);

        if (b.fillBond > 0) {
            _pull(b.fillBond);
        }
        emit BidReserved(bidId, msg.sender, b.fillDeadline);
    }

    /// @notice Collect the escrow (plus the fill bond back) by revealing the
    ///         32-byte secret the maker published on Midstate.
    /// @dev    Callable by anyone before the fill deadline: the tokens always
    ///         go to the reservation's holder.
    function claimBid(bytes32 bidId, bytes32 preimage) external nonReentrant {
        Bid storage b = bids[bidId];
        require(b.amount > 0, "not found");
        require(!b.settled, "settled");
        require(b.filler != address(0), "not reserved");
        require(block.timestamp <= b.fillDeadline, "fill lapsed");
        require(blake3_256(preimage) == b.hashlock, "bad preimage");

        b.settled = true;
        emit BidClaimed(bidId, b.hashlock, preimage);
        _push(b.filler, b.amount + b.fillBond);
    }

    /// @notice Cancel a bid and take the escrow back. Maker only. Instant while
    ///         the bid is unreserved; blocked during a live reservation; allowed
    ///         again once the reservation lapses, sweeping the forfeited bond.
    function cancelBid(bytes32 bidId) external nonReentrant {
        Bid storage b = bids[bidId];
        require(b.amount > 0, "not found");
        require(!b.settled, "settled");
        require(msg.sender == b.maker, "not maker");
        require(b.filler == address(0) || block.timestamp > b.fillDeadline, "reserved");

        b.settled = true;
        emit BidCancelled(bidId);
        _push(b.maker, b.amount + (b.filler != address(0) ? b.fillBond : 0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Moving the token
    // ─────────────────────────────────────────────────────────────────────────

    /// Uses the caller's permit, if it still stands. A permit copied out of
    /// the mempool and used first has already set the allowance it signs, so
    /// a failed permit is passed over: the transfer that follows is what
    /// decides.
    function _permit(uint256 amount, uint256 deadline, uint8 v, bytes32 r, bytes32 s) private {
        try IERC20Permit(address(token)).permit(msg.sender, address(this), amount, deadline, v, r, s) {} catch {}
    }

    /// Takes `amount` from the caller, refusing a token that delivers less.
    function _pull(uint256 amount) private {
        uint256 before = token.balanceOf(address(this));
        _call(abi.encodeWithSelector(IERC20.transferFrom.selector, msg.sender, address(this), amount));
        require(token.balanceOf(address(this)) - before == amount, "token delivered less");
    }

    function _push(address to, uint256 amount) private {
        _call(abi.encodeWithSelector(IERC20.transfer.selector, to, amount));
    }

    /// A token call that must succeed: one that reverts, or returns false,
    /// stops the transaction.
    function _call(bytes memory data) private {
        (bool ok, bytes memory ret) = address(token).call(data);
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "token transfer failed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  BLAKE3 (single 32-byte block, root) — matches Midstate core::types::hash
    //  The same code as MidstateAtomicSwap's.
    // ─────────────────────────────────────────────────────────────────────────

    uint256 private constant M = 0xFFFFFFFF;

    function _rotr(uint256 x, uint256 n) private pure returns (uint256) {
        return ((x >> n) | (x << (32 - n))) & M;
    }

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
        uint256[] memory m = new uint256[](16);
        for (uint256 i = 0; i < 8; i++) {
            uint256 base = 4 * i;
            uint256 b0 = uint256(uint8(preimage[base]));
            uint256 b1 = uint256(uint8(preimage[base + 1]));
            uint256 b2 = uint256(uint8(preimage[base + 2]));
            uint256 b3 = uint256(uint8(preimage[base + 3]));
            m[i] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) & M;
        }

        uint256[] memory st = new uint256[](16);
        st[0] = 0x6A09E667; st[1] = 0xBB67AE85; st[2] = 0x3C6EF372; st[3] = 0xA54FF53A;
        st[4] = 0x510E527F; st[5] = 0x9B05688C; st[6] = 0x1F83D9AB; st[7] = 0x5BE0CD19;
        st[8] = 0x6A09E667; st[9] = 0xBB67AE85; st[10] = 0x3C6EF372; st[11] = 0xA54FF53A;
        st[12] = 0;         st[13] = 0;         st[14] = 32;        st[15] = 11;

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

        bytes memory outb = new bytes(32);
        for (uint256 i = 0; i < 8; i++) {
            uint256 w = (st[i] ^ st[i + 8]) & M;
            outb[4 * i]     = bytes1(uint8(w & 0xff));
            outb[4 * i + 1] = bytes1(uint8((w >> 8) & 0xff));
            outb[4 * i + 2] = bytes1(uint8((w >> 16) & 0xff));
            outb[4 * i + 3] = bytes1(uint8((w >> 24) & 0xff));
        }
        bytes32 res;
        assembly {
            res := mload(add(outb, 0x20))
        }
        return res;
    }
}
