'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { ConnectionProvider, WalletProvider, useWallet } from '@solana/wallet-adapter-react';
import { WalletModalProvider, WalletMultiButton } from '@solana/wallet-adapter-react-ui';
import { PhantomWalletAdapter } from '@solana/wallet-adapter-wallets';
import { Connection, PublicKey } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import SolanaSwapIdl from '../idl/solana_swap.json';
import * as blake3 from 'blake3/browser';

import '@solana/wallet-adapter-react-ui/styles.css';

const RELAYER_URL = "http://localhost:3001/api";
const MIDSTATE_RPC = "http://127.0.0.1:8545"; 
const PROGRAM_ID = new PublicKey(SolanaSwapIdl.metadata.address);

function SwapApp() {
    const { publicKey, signTransaction } = useWallet();
    const [orders, setOrders] = useState<any[]>([]);
    const [myMidstateAddress, setMyMidstateAddress] = useState("");

    useEffect(() => {
        fetchOrders();
        const int = setInterval(fetchOrders, 5000);
        return () => clearInterval(int);
    }, []);

    const fetchOrders = async () => {
        try {
            const res = await fetch(`${RELAYER_URL}/orders`);
            setOrders(await res.json());
        } catch(e) {}
    };

    const createOrder = async () => {
        if (!publicKey || !myMidstateAddress) return alert("Connect Solana wallet and enter Midstate address");
        await fetch(`${RELAYER_URL}/orders`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                makerSolanaAddress: publicKey.toString(),
                makerMidstateAddress: myMidstateAddress,
                midstateAmount: 100, 
                timeAmount: 500
            })
        });
        fetchOrders();
    };

    const fillOrder = async (order: any) => {
        if (!publicKey || !signTransaction) return alert("Please connect Solana Wallet");

        // 1. Generate 32 bytes of secure random data
        const preimage = new Uint8Array(32);
        crypto.getRandomValues(preimage);
        
        // 2. Execute real BLAKE3 hash to align perfectly with Midstate OP_HASH
        const secretHashArray = blake3.hash(preimage);
        const secretHash = new Uint8Array(secretHashArray);

        const connection = new Connection("https://api.devnet.solana.com");
        const provider = new AnchorProvider(connection, { publicKey, signTransaction } as any, {});
        const program = new Program(SolanaSwapIdl as any, PROGRAM_ID, provider);

        const [escrowPda] = PublicKey.findProgramAddressSync(
            [Buffer.from("escrow"), publicKey.toBuffer(), Buffer.from(secretHash)],
            program.programId
        );

        try {
            console.log("Locking $TIME on Solana...");
            const timeout = Math.floor(Date.now() / 1000) + 86400; // 24 hours
            
            await program.methods.initializeEscrow(
                new (provider.connection as any).BN(order.timeAmount), 
                Array.from(secretHash), 
                new (provider.connection as any).BN(timeout)
            )
            .accounts({
                maker: publicKey,
                taker: new PublicKey(order.makerSolanaAddress),
                escrowState: escrowPda,
            })
            .rpc();

            await fetch(`${RELAYER_URL}/orders/${order.id}/solana-locked`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    secretHash: Buffer.from(secretHash).toString('hex'),
                    escrowPda: escrowPda.toString()
                })
            });

            alert(`Solana Locked! Secret Preimage: ${Buffer.from(preimage).toString('hex')}. Save this to claim your Midstate coins!`);
        } catch (e) {
            console.error(e);
            alert("Failed to lock Solana");
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-10 font-sans">
            <div className="max-w-4xl mx-auto">
                <div className="flex justify-between items-center mb-10">
                    <h1 className="text-3xl font-bold text-orange-500">Midstate Atomic DEX</h1>
                    <WalletMultiButton />
                </div>

                <div className="bg-gray-800 p-6 rounded-lg mb-10">
                    <h2 className="text-xl font-semibold mb-4">Create Limit Order</h2>
                    <input 
                        type="text" 
                        placeholder="Your Midstate Address" 
                        className="w-full p-3 bg-gray-700 rounded text-white mb-4"
                        value={myMidstateAddress}
                        onChange={e => setMyMidstateAddress(e.target.value)}
                    />
                    <button onClick={createOrder} className="bg-orange-500 px-6 py-3 rounded font-bold hover:bg-orange-600">
                        Sell 100 Midstate for 500 $TIME
                    </button>
                </div>

                <h2 className="text-xl font-semibold mb-4">Order Book</h2>
                <div className="space-y-4">
                    {orders.map(o => (
                        <div key={o.id} className="bg-gray-800 p-4 rounded flex justify-between items-center border border-gray-700">
                            <div>
                                <p className="font-mono text-sm text-gray-400">Maker: {o.makerMidstateAddress.substring(0,16)}...</p>
                                <p className="text-lg">Buy <span className="text-orange-500 font-bold">{o.midstateAmount} MID</span> for <span className="text-green-400 font-bold">{o.timeAmount} TIME</span></p>
                            </div>
                            <button onClick={() => fillOrder(o)} className="bg-green-500 px-6 py-2 rounded font-bold hover:bg-green-600 text-gray-900">
                                Fill Order
                            </button>
                        </div>
                    ))}
                    {orders.length === 0 && <p className="text-gray-500 italic">No open orders.</p>}
                </div>
            </div>
        </div>
    );
}

export default function Home() {
    const endpoint = useMemo(() => "https://api.devnet.solana.com", []);
    const wallets = useMemo(() => [new PhantomWalletAdapter()], []);

    return (
        <ConnectionProvider endpoint={endpoint}>
            <WalletProvider wallets={wallets} autoConnect>
                <WalletModalProvider>
                    <SwapApp />
                </WalletModalProvider>
            </WalletProvider>
        </ConnectionProvider>
    );
}
