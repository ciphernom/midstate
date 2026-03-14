import express from 'express';
import cors from 'cors';
import { Connection, Keypair, PublicKey, ComputeBudgetProgram, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import { PrismaClient } from '@prisma/client';
import * as dotenv from 'dotenv';
import SolanaSwapIdl from './idl/solana_swap.json';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
const prisma = new PrismaClient();

const MIDSTATE_RPC = process.env.MIDSTATE_RPC_URL || "http://127.0.0.1:8545";
const SOLANA_RPC = process.env.SOLANA_RPC_URL || "https://api.devnet.solana.com"; // Using Devnet for safety
const PROGRAM_ID = new PublicKey(SolanaSwapIdl.metadata.address);

const connection = new Connection(SOLANA_RPC, 'confirmed');

// Automatically use the default Solana CLI wallet as the Cranker/Relayer
const fs = require('fs');
const os = require('os');
const keypairFile = fs.readFileSync(`${os.homedir()}/.config/solana/id.json`);
const crankerKeypair = Keypair.fromSecretKey(Uint8Array.from(JSON.parse(keypairFile)));

const provider = new AnchorProvider(connection, new Wallet(crankerKeypair), {});
const program = new Program(SolanaSwapIdl as any, PROGRAM_ID, provider);

app.post('/api/orders', async (req, res) => {
    const order = await prisma.swapOrder.create({
        data: {
            makerMidstateAddress: req.body.makerMidstateAddress,
            makerSolanaAddress: req.body.makerSolanaAddress,
            midstateAmount: BigInt(req.body.midstateAmount),
            timeAmount: BigInt(req.body.timeAmount),
            status: "OPEN"
        }
    });
    res.json({ ...order, midstateAmount: order.midstateAmount.toString(), timeAmount: order.timeAmount.toString() });
});

app.get('/api/orders', async (req, res) => {
    const orders = await prisma.swapOrder.findMany({ where: { status: "OPEN" } });
    res.json(orders.map(o => ({
        ...o,
        midstateAmount: o.midstateAmount.toString(),
        timeAmount: o.timeAmount.toString()
    })));
});

app.post('/api/orders/:id/solana-locked', async (req, res) => {
    const order = await prisma.swapOrder.update({
        where: { id: req.params.id },
        data: { status: "LOCKED_ON_SOL", secretHash: req.body.secretHash, escrowPda: req.body.escrowPda }
    });
    res.json({ status: order.status });
});

app.post('/api/orders/:id/midstate-locked', async (req, res) => {
    const order = await prisma.swapOrder.update({
        where: { id: req.params.id },
        data: { status: "LOCKED_ON_MIDSTATE", htlcCoinId: req.body.htlcCoinId }
    });
    res.json({ status: order.status });
});

let lastScannedHeight = 0;

async function watchMidstateMempool() {
    console.log("👀 Relayer started. Watching Midstate Mempool & Blocks...");
    try {
        const stateRes = await fetch(`${MIDSTATE_RPC}/state`);
        const state = await stateRes.json();
        lastScannedHeight = state.height;
    } catch (e) {
        console.log("Could not fetch initial Midstate height. Retrying...");
    }

    setInterval(async () => {
        try {
            const stateRes = await fetch(`${MIDSTATE_RPC}/state`);
            const state = await stateRes.json();
            const currentHeight = state.height;

            const activeSwaps = await prisma.swapOrder.findMany({
                where: { status: "LOCKED_ON_MIDSTATE" }
            });

            if (activeSwaps.length === 0) return;

            const mempoolRes = await fetch(`${MIDSTATE_RPC}/mempool`);
            const mempool = await mempoolRes.json();
            await extractAndExecute(activeSwaps, mempool.transactions);

            if (lastScannedHeight > 0 && lastScannedHeight < currentHeight) {
                for (let h = lastScannedHeight + 1; h <= currentHeight; h++) {
                    const blockRes = await fetch(`${MIDSTATE_RPC}/block/${h}`);
                    const block = await blockRes.json();
                    await extractAndExecute(activeSwaps, block.transactions);
                }
            }
            lastScannedHeight = currentHeight;
        } catch (err) { }
    }, 2000);
}

async function extractAndExecute(activeSwaps: any[], transactions: any[]) {
    for (const swap of activeSwaps) {
        for (const tx of transactions) {
            if (!tx.Reveal) continue;

            const inputIndex = tx.Reveal.inputs.findIndex((i: any) => i.coin_id === swap.htlcCoinId);
            if (inputIndex === -1) continue;

            // Fetch the UNSTRIPPED transaction using the dedicated endpoint we added
            const fullTxRes = await fetch(`${MIDSTATE_RPC}/tx/by_input`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ coin_id: swap.htlcCoinId }) 
            });
            if (!fullTxRes.ok) continue;
            
            const fullTx = await fullTxRes.json();
            const witnessStack = fullTx.Reveal.witnesses[inputIndex].ScriptInputs;
            
            const pathIndicator = witnessStack[witnessStack.length - 1];
            const isTrue = pathIndicator !== "00" && pathIndicator !== "";
            if (!isTrue) continue;

            const preimageHex = witnessStack[1]; 
            const preimageBuffer = Buffer.from(preimageHex, 'hex');

            console.log(`🚨 PREIMAGE FOUND FOR SWAP ${swap.id}: ${preimageHex}`);
            
            const updated = await prisma.swapOrder.updateMany({
                where: { id: swap.id, status: "LOCKED_ON_MIDSTATE" },
                data: { status: "CLAIMING" }
            });

            if (updated.count > 0) {
                await executeSolanaClaimReliably(swap, preimageBuffer);
            }
        }
    }
}

async function executeSolanaClaimReliably(swap: any, preimage: Buffer) {
    console.log(`⚡ Building Solana Claim Tx...`);
    try {
        const claimIx = await program.methods.claim(preimage)
            .accounts({
                cranker: crankerKeypair.publicKey,
                escrowState: new PublicKey(swap.escrowPda),
                taker: new PublicKey(swap.makerSolanaAddress),
            })
            .instruction();

        const addPriorityFee = ComputeBudgetProgram.setComputeUnitPrice({ microLamports: 200_000 });
        const tx = new Transaction().add(addPriorityFee, claimIx);
        tx.recentBlockhash = (await connection.getLatestBlockhash('confirmed')).blockhash;
        tx.feePayer = crankerKeypair.publicKey;
        tx.sign(crankerKeypair);
        
        const txSignature = await connection.sendRawTransaction(tx.serialize(), { skipPreflight: true });
        console.log(`✅ Swap Complete! Solana Tx: ${txSignature}`);
        
        await prisma.swapOrder.update({ where: { id: swap.id }, data: { status: "COMPLETED" } });
    } catch (err) {
        console.error("Fatal Error executing Solana claim:", err);
        await prisma.swapOrder.update({ where: { id: swap.id }, data: { status: "FAILED" } });
    }
}

app.listen(3001, () => {
    console.log("🚀 Midstate Relayer running on port 3001");
    watchMidstateMempool();
});
