// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// A stand-in for USDC on Base, for tests only: six decimals, EIP-2612 permit
/// under the domain FiatToken uses ("USD Coin", version "2"), a blacklist,
/// and two misbehaviours to switch on: a fee taken on transfer, and a
/// transfer that returns nothing.
contract MockUSDC {
    string public constant name = "USD Coin";
    string public constant symbol = "USDC";
    uint8 public constant decimals = 6;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public nonces;
    mapping(address => bool) public blacklisted;
    uint256 public feeBps;
    bool public silent;

    bytes32 public constant PERMIT_TYPEHASH =
        keccak256("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)");

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function DOMAIN_SEPARATOR() public view returns (bytes32) {
        return keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes(name)),
                keccak256(bytes("2")),
                block.chainid,
                address(this)
            )
        );
    }

    function mint(address to, uint256 v) external {
        balanceOf[to] += v;
        totalSupply += v;
        emit Transfer(address(0), to, v);
    }

    function setFee(uint256 bps) external { feeBps = bps; }
    function setSilent(bool on) external { silent = on; }
    function setBlacklisted(address who, bool on) external { blacklisted[who] = on; }

    function approve(address spender, uint256 v) external returns (bool) {
        allowance[msg.sender][spender] = v;
        emit Approval(msg.sender, spender, v);
        return true;
    }

    function permit(address owner, address spender, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external {
        require(block.timestamp <= deadline, "permit expired");
        bytes32 digest = keccak256(
            abi.encodePacked(
                "\x19\x01",
                DOMAIN_SEPARATOR(),
                keccak256(abi.encode(PERMIT_TYPEHASH, owner, spender, value, nonces[owner]++, deadline))
            )
        );
        address signer = ecrecover(digest, v, r, s);
        require(signer != address(0) && signer == owner, "invalid signature");
        allowance[owner][spender] = value;
        emit Approval(owner, spender, value);
    }

    function _move(address from, address to, uint256 v) private {
        require(!blacklisted[from] && !blacklisted[to], "blacklisted");
        require(balanceOf[from] >= v, "balance");
        balanceOf[from] -= v;
        uint256 fee = v * feeBps / 10_000;
        balanceOf[to] += v - fee;
        totalSupply -= fee;
        emit Transfer(from, to, v - fee);
    }

    function transfer(address to, uint256 v) external returns (bool) {
        _move(msg.sender, to, v);
        if (silent) assembly { return(0, 0) }
        return true;
    }

    function transferFrom(address from, address to, uint256 v) external returns (bool) {
        require(allowance[from][msg.sender] >= v, "allowance");
        allowance[from][msg.sender] -= v;
        _move(from, to, v);
        if (silent) assembly { return(0, 0) }
        return true;
    }
}

/// A token that answers `false` rather than reverting, as some do.
contract FalseToken {
    mapping(address => uint256) public balanceOf;
    function mint(address to, uint256 v) external { balanceOf[to] += v; }
    function approve(address, uint256) external pure returns (bool) { return true; }
    function transfer(address, uint256) external pure returns (bool) { return false; }
    function transferFrom(address, address, uint256) external pure returns (bool) { return false; }
}
