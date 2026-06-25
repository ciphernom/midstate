// SPDX-License-Identifier: GPL
pragma solidity ^0.8.0;

contract MidstateAtomicSwap {
    struct Swap {
        address payable maker;
        address payable taker;
        bytes32 expectedKeccak;
        uint256 amount;
        uint256 timeout;
        bool claimed;
    }

    mapping(bytes32 => Swap) public swaps; 

    function lockFunds(bytes32 _expectedKeccak, address payable _maker, uint256 _timeoutBlocks) external payable {
        require(swaps[_expectedKeccak].amount == 0, "Swap already exists");
        swaps[_expectedKeccak] = Swap({
            maker: _maker,
            taker: payable(msg.sender),
            expectedKeccak: _expectedKeccak,
            amount: msg.value,
            timeout: block.number + _timeoutBlocks,
            claimed: false
        });
    }

    function claim(bytes calldata _secret) external {
        bytes32 expected = keccak256(_secret);
        Swap storage s = swaps[expected];
        require(s.amount > 0, "Swap not found");
        require(!s.claimed, "Already claimed");
        
        s.claimed = true;
        
        // Modern gas-safe transfer for Maker
        (bool success, ) = s.maker.call{value: s.amount}("");
        require(success, "Transfer to Maker failed");
    }

    function refund(bytes32 _expectedKeccak) external {
        Swap storage s = swaps[_expectedKeccak];
        require(s.amount > 0, "Swap not found");
        require(!s.claimed, "Already claimed");
        require(block.number >= s.timeout, "Timeout not reached");

        s.claimed = true;
        
        // Modern gas-safe transfer for Taker refund
        (bool success, ) = s.taker.call{value: s.amount}("");
        require(success, "Refund to Taker failed");
    }
}
