import fs from 'fs/promises';
import path from 'path';

export class StorageProvider {
    async saveMetadata(dataStr) { throw new Error("Not implemented"); }
    async loadMetadata() { throw new Error("Not implemented"); }
    async saveMssTree(addressHex, uint8Array) { throw new Error("Not implemented"); }
    async loadMssTree(addressHex) { throw new Error("Not implemented"); }
}

export class MemoryStorage extends StorageProvider {
    constructor() {
        super();
        this.metadata = null;
        this.mssTrees = new Map();
    }
    async saveMetadata(dataStr) { this.metadata = dataStr; }
    async loadMetadata() { return this.metadata; }
    async saveMssTree(addressHex, uint8Array) { this.mssTrees.set(addressHex, uint8Array); }
    async loadMssTree(addressHex) { return this.mssTrees.get(addressHex); }
}

export class NodeFSStorage extends StorageProvider {
    constructor(dirPath = './midstate_wallet_data') {
        super();
        this.dirPath = dirPath;
    }
    async _ensureDir() { await fs.mkdir(this.dirPath, { recursive: true }); }
    async saveMetadata(dataStr) {
        await this._ensureDir();
        await fs.writeFile(path.join(this.dirPath, 'wallet.json'), dataStr);
    }
    async loadMetadata() {
        try { return await fs.readFile(path.join(this.dirPath, 'wallet.json'), 'utf8'); } 
        catch (e) { return null; }
    }
    async saveMssTree(addressHex, uint8Array) {
        await this._ensureDir();
        await fs.writeFile(path.join(this.dirPath, `mss_${addressHex}.bin`), uint8Array);
    }
    async loadMssTree(addressHex) {
        try { return await fs.readFile(path.join(this.dirPath, `mss_${addressHex}.bin`)); } 
        catch (e) { return null; }
    }
}

export class BrowserStorage extends StorageProvider {
    async saveMetadata(dataStr) { localStorage.setItem('midstate_wallet_meta', dataStr); }
    async loadMetadata() { return localStorage.getItem('midstate_wallet_meta'); }
    async _openIDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open('midstate_sdk_db', 1);
            req.onupgradeneeded = () => {
                if (!req.result.objectStoreNames.contains('mss_trees')) req.result.createObjectStore('mss_trees');
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }
    async saveMssTree(addressHex, uint8Array) {
        const db = await this._openIDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('mss_trees', 'readwrite');
            tx.objectStore('mss_trees').put(uint8Array, `mss_${addressHex}`);
            tx.oncomplete = () => { db.close(); resolve(); };
            tx.onerror = () => { db.close(); reject(tx.error); };
        });
    }
    async loadMssTree(addressHex) {
        const db = await this._openIDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('mss_trees', 'readonly');
            const req = tx.objectStore('mss_trees').get(`mss_${addressHex}`);
            req.onsuccess = () => { db.close(); resolve(req.result); };
            req.onerror = () => { db.close(); reject(req.error); };
        });
    }
}
