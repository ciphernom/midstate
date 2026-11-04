//! The launchpad contract (sdk/src/launchpad.js) on the node's own VM.
//!
//! `sdk/test-launchpad.mjs` runs every launch spend through the SDK's VM, which
//! compares signatures as strings. It also records each spend with real MSS
//! signatures over a fixed commitment (`WRITE_VECTORS=1 node test-launchpad.mjs`).
//! This replays them through `execute_script`, once per reserve input as
//! consensus does, and requires the same verdict: every honest trade accepted,
//! every attack refused. A disagreement means the SDK VM has drifted from this
//! one, and the launcher would build trades the chain treats differently.

use midstate::core::script::{execute_script, ExecContext};
use midstate::core::{hash, OutputData};
use serde_json::Value;

const VECTORS: &str = include_str!("../sdk/test-vectors/launchpad.json");

fn bytes(v: &Value) -> Vec<u8> {
    hex::decode(v.as_str().expect("hex string")).expect("valid hex")
}

fn b32(v: &Value) -> [u8; 32] {
    bytes(v).try_into().expect("32 bytes")
}

fn want_ok(v: &Value) -> bool {
    v["ok"].as_bool().unwrap()
}

#[test]
fn launchpad_spends_get_the_same_verdict_on_the_node_vm() {
    let file: Value = serde_json::from_str(VECTORS).expect("vectors parse");
    let height = file["height"].as_u64().unwrap();
    let commitment = b32(&file["commitment"]);
    let code = bytes(&file["code"]);
    // The code copy travels as two witness items (see MAX_PLAIN_ITEM in launchpad.js).
    let split = file["code_split"].as_u64().unwrap() as usize;
    let (code_a, code_b) = code.split_at(split);
    let sigs: Vec<Vec<u8>> = file["sigs"].as_array().unwrap().iter().map(bytes).collect();
    let vectors = file["vectors"].as_array().unwrap();
    assert!(vectors.len() > 30, "too few vectors: {}", vectors.len());

    let mut accepted = 0;
    let mut refused = 0;
    for v in vectors {
        let name = v["name"].as_str().unwrap();

        // PUSH_DATA(56) state DROP code
        let mut bytecode = vec![0x01, 0x38, 0x00];
        bytecode.extend(bytes(&v["state"]));
        bytecode.push(0x10);
        bytecode.extend(&code);
        let this_address = hash(&bytecode);

        let witness: Vec<Vec<u8>> = v["witness"].as_array().unwrap().iter().map(|w| {
            let s = w.as_str().unwrap();
            if s == "$codeA" {
                code_a.to_vec()
            } else if s == "$codeB" {
                code_b.to_vec()
            } else if let Some(i) = s.strip_prefix("$sig") {
                sigs[i.parse::<usize>().unwrap()].clone()
            } else {
                hex::decode(s).unwrap()
            }
        }).collect();

        let outputs: Vec<OutputData> = v["outputs"].as_array().unwrap().iter().map(|o| {
            if let Some(s) = o.get("standard") {
                OutputData::Standard { address: b32(&s["address"]), value: s["value"].as_u64().unwrap(), salt: b32(&s["salt"]) }
            } else {
                let c = &o["confidential"];
                OutputData::Confidential { address: b32(&c["address"]), commitment: b32(&c["commitment"]), salt: b32(&c["salt"]) }
            }
        }).collect();

        // Only the trader's signature may parse as an MSS signature: anything
        // else would be recorded as a spent leaf by the first trade's block and
        // make every later trade look like key reuse.
        if want_ok(v) {
            let parsed = midstate::core::mss::mss_sigs_in_stack(&witness).len();
            assert_eq!(parsed, 1, "{}: {} witness items parse as MSS signatures", name, parsed);
            assert_ne!(witness[0].len(), midstate::core::wots::SIG_SIZE, "{}: first item reads as WOTS", name);
        }

        let values: Vec<u64> = v["inputs"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        let sum: u64 = values.iter().sum();
        let mut verdict = Ok(());
        for &value in &values {
            let ctx = ExecContext {
                commitment: &commitment,
                height,
                outputs: &outputs,
                input_value: value,
                input_state: None,
                this_address,
                sum_input_value: sum,
            };
            if let Err(e) = execute_script(&bytecode, &witness, &ctx) {
                verdict = Err(e);
                break;
            }
        }

        let want = want_ok(v);
        match (&verdict, want) {
            (Ok(()), true) => accepted += 1,
            (Err(_), false) => refused += 1,
            (Ok(()), false) => panic!("node VM ACCEPTED a spend the SDK refuses: {}", name),
            (Err(e), true) => panic!("node VM refused an honest spend ({:?}): {}", e, name),
        }
    }
    assert!(accepted >= 10 && refused >= 20, "accepted {}, refused {}", accepted, refused);
}

#[test]
fn launchpad_needs_the_v6_opcodes() {
    // One height below activation, the same honest spend is invalid: the
    // launcher page must not offer launches before then.
    let file: Value = serde_json::from_str(VECTORS).unwrap();
    let code = bytes(&file["code"]);
    let v = file["vectors"].as_array().unwrap().iter().find(|v| v["ok"].as_bool() == Some(true)).unwrap();
    let mut bytecode = vec![0x01, 0x38, 0x00];
    bytecode.extend(bytes(&v["state"]));
    bytecode.push(0x10);
    bytecode.extend(&code);
    let height = file["height"].as_u64().unwrap();
    assert_eq!(height, midstate::core::types::V6_ACTIVATION_HEIGHT);
    assert!(midstate::core::script::validate_structure(&bytecode, height - 1).is_err());
    assert!(midstate::core::script::validate_structure(&bytecode, height).is_ok());
}
