//! The burn index over HTTP (`storage::burn_index`, `core::burns`): what the
//! chain burns against what it issues, and MSB-1 backing by subject.
//!
//! * `GET /burns`: totals, each year's issuance and burn by kind, the last
//!   365 days, and all the backing held, with the share of issuance keeping
//!   it up burns.
//! * `GET /burns/days?days=`: burned and issued by day, newest last.
//! * `GET /burns/top?purpose=&limit=`: the most-backed subjects.
//! * `GET /burns/recent?limit=`: the newest MSB-1 burns.
//! * `GET /burns/:purpose/:subject?limit=`: a subject's backing and burns.
//! * `GET /burns/notary/:hash`: a document hash, under both notary purposes.

use crate::core::burns::{self, Backing, Purpose};
use crate::core::types::{block_reward, BLOCKS_PER_YEAR};
use crate::node::NodeHandle;
use crate::storage::burn_index::{self, Totals, DAY};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::collections::HashMap;

type AppState = NodeHandle;

fn error(code: StatusCode, e: impl std::fmt::Display) -> Response {
    (code, Json(json!({ "error": e.to_string() }))).into_response()
}

fn unavailable(e: anyhow::Error) -> Response {
    error(StatusCode::SERVICE_UNAVAILABLE, e)
}

fn limit(q: &HashMap<String, String>, default: usize, max: usize) -> usize {
    q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(default).clamp(1, max)
}

/// Burns by kind, and backing by purpose.
pub fn totals_json(t: &Totals) -> Value {
    let mut by_purpose = serde_json::Map::new();
    for p in Purpose::ALL {
        by_purpose.insert(p.word(), t.backing[p.0 as usize].into());
    }
    if t.backing[0] > 0 {
        by_purpose.insert("other".into(), t.backing[0].into());
    }
    json!({
        "burned": t.burned(),
        "names": t.names,
        "backing": t.backing_total(),
        "dex": t.dex,
        "other": t.other,
        "by_purpose": by_purpose,
        "count": t.count,
        "backing_count": t.backing_count,
    })
}

fn share(burned: u64, issued: u64) -> Value {
    if issued == 0 { Value::Null } else { (burned as f64 / issued as f64).into() }
}

fn year_json(y: &burn_index::Year) -> Value {
    json!({
        "year": if y.year == u64::MAX { Value::Null } else { (y.year + 1).into() },
        "from": y.from,
        "to": y.to,
        "issued": y.issued,
        "burned": y.burned.burned(),
        "share": share(y.burned.burned(), y.issued),
        "by_kind": totals_json(&y.burned),
    })
}

fn index_json(m: &burn_index::Meta) -> Value {
    json!({ "height": m.next, "chain": m.chain_len, "ready": m.ready(), "from": m.from, "missing": m.missing.checked_sub(1) })
}

/// `GET /burns`.
pub async fn get_burns(State(node): State<AppState>) -> Response {
    let db = node.storage.db();
    let m = match burn_index::meta(db) {
        Ok(m) => m,
        Err(e) => return unavailable(e),
    };
    let (years, trailing) = match burn_index::years(db) {
        Ok(y) => y,
        Err(e) => return unavailable(e),
    };
    let held = burn_index::held(db).unwrap_or_default();
    let blocks: f64 = held.iter().map(|(_, b, _)| b).sum();
    let mut by_purpose = serde_json::Map::new();
    for (p, b, n) in &held {
        by_purpose.insert(p.word(), json!({ "blocks": b, "subjects": n }));
    }
    Json(json!({
        "index": index_json(&m),
        "reward": block_reward(m.next),
        "half_life": burns::HALF_LIFE,
        "totals": totals_json(&m.totals),
        "years": years.iter().map(year_json).collect::<Vec<_>>(),
        "trailing": year_json(&trailing),
        "held": {
            "blocks": blocks,
            "text": burns::blocks_text(blocks),
            // What keeping it all up burns, as a share of each year's issuance.
            "upkeep_share": burns::upkeep(blocks) / BLOCKS_PER_YEAR as f64,
            "subjects": held.iter().map(|(_, _, n)| n).sum::<u64>(),
            "by_purpose": by_purpose,
        },
        "tiers": burns::TIERS.iter().map(|(b, w)| json!({ "blocks": b, "name": w })).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// `GET /burns/days?days=90`.
pub async fn get_burn_days(State(node): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let db = node.storage.db();
    let m = match burn_index::meta(db) {
        Ok(m) => m,
        Err(e) => return unavailable(e),
    };
    let n: u64 = q.get("days").and_then(|v| v.parse().ok()).unwrap_or(90).clamp(1, 3_650);
    let last = m.next.saturating_sub(1) / DAY;
    let first = (last + 1).saturating_sub(n);
    let have: HashMap<u64, Totals> = match burn_index::days(db, first, last + 1) {
        Ok(d) => d.into_iter().collect(),
        Err(e) => return unavailable(e),
    };
    let rows: Vec<Value> = (first..=last)
        .map(|d| {
            let t = have.get(&d).cloned().unwrap_or_default();
            let (from, to) = (d * DAY, ((d + 1) * DAY).min(m.next));
            let issued = burns::issued(from.max(m.from), to);
            json!({ "day": d, "from": from, "to": to, "issued": issued, "burned": t.burned(), "share": share(t.burned(), issued), "by_kind": totals_json(&t) })
        })
        .collect();
    Json(json!({ "index": index_json(&m), "days": rows })).into_response()
}

fn purpose_of(s: &str) -> Result<Purpose, Response> {
    Purpose::parse(s).ok_or_else(|| error(StatusCode::BAD_REQUEST, format!("no purpose {s:?}: name, key, evm, pool, hub, notary or notary-sha256")))
}

/// `GET /burns/top?purpose=name&limit=50`.
pub async fn get_burn_top(State(node): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let p = match purpose_of(q.get("purpose").map(String::as_str).unwrap_or("name")) {
        Ok(p) => p,
        Err(r) => return r,
    };
    match burn_index::top(node.storage.db(), p, limit(&q, 50, 500)) {
        Ok(list) => Json(json!({ "purpose": p.word(), "top": list.iter().map(|s| s.json()).collect::<Vec<_>>() })).into_response(),
        Err(e) => unavailable(e),
    }
}

/// `GET /burns/recent?limit=50`.
pub async fn get_burn_recent(State(node): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    match burn_index::recent(node.storage.db(), limit(&q, 50, 500), 20_000) {
        Ok(list) => Json(
            list.iter()
                .map(|r| {
                    json!({
                        "height": r.height,
                        "tx": r.tx,
                        "time": r.time,
                        "value": r.value,
                        "weight": burns::weight(r.value, r.height),
                        "purpose": r.purpose.word(),
                        "subject": burns::subject_text(r.purpose, &r.subject),
                        "subject_hex": hex::encode(r.subject),
                        "label": r.label,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(e) => unavailable(e),
    }
}

/// A subject's backing and burns, as JSON.
pub fn subject_json(db: &redb::Database, p: Purpose, subject: &[u8; 32], limit: usize) -> anyhow::Result<Value> {
    let s = burn_index::backing(db, p, subject, None)?;
    let list = burn_index::history(db, p, subject, limit)?;
    Ok(json!({
        "purpose": p.word(),
        "subject": burns::subject_text(p, subject),
        "subject_hex": hex::encode(subject),
        "backing": s.as_ref().map(|s| s.json()),
        "burns": list.iter().map(|b| json!({
            "height": b.height, "tx": b.tx, "time": b.time, "value": b.value,
            "weight": b.weight, "weight_now": b.weight_now, "label": b.label,
        })).collect::<Vec<_>>(),
    }))
}

/// What a name's backing looks like inside `GET /name/:name`.
pub fn name_backing(db: &redb::Database, name: &str) -> Value {
    let Ok(subject) = burns::name_subject(name) else { return Value::Null };
    match burn_index::backing(db, Purpose::NAME, &subject, None) {
        Ok(Some(s)) => s.json(),
        _ => Value::Null,
    }
}

/// `GET /burns/:purpose/:subject?limit=100`.
pub async fn get_burn_subject(
    State(node): State<AppState>,
    Path((purpose, subject)): Path<(String, String)>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let p = match purpose_of(&purpose) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let s = match burns::parse_subject(p, &subject) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    match subject_json(node.storage.db(), p, &s, limit(&q, 100, 1_000)) {
        Ok(v) => Json(v).into_response(),
        Err(e) => unavailable(e),
    }
}

/// `GET /burns/notary/:hash`: a document's hash under both notary purposes.
pub async fn get_burn_notary(State(node): State<AppState>, Path(hash): Path<String>) -> Response {
    let s = match burns::parse_subject(Purpose::NOTARY, &hash) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let db = node.storage.db();
    let both = (subject_json(db, Purpose::NOTARY, &s, 100), subject_json(db, Purpose::NOTARY_SHA256, &s, 100));
    match both {
        (Ok(a), Ok(b)) => Json(json!({ "hash": hex::encode(s), "blake3": a, "sha256": b })).into_response(),
        (Err(e), _) | (_, Err(e)) => unavailable(e),
    }
}

/// The payload of an MSB-1 burn, for clients that build their own
/// transactions: `GET /burns/payload?purpose=&subject=&label=`.
pub async fn get_burn_payload(Query(q): Query<HashMap<String, String>>) -> Response {
    let p = match purpose_of(q.get("purpose").map(String::as_str).unwrap_or("")) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let s = match burns::parse_subject(p, q.get("subject").map(String::as_str).unwrap_or("")) {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    match Backing::new(p, s, q.get("label").map(String::as_str).unwrap_or("")) {
        Ok(b) => Json(json!({ "payload": hex::encode(b.payload()) })).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_routes_fit_together() {
        async fn ok() -> &'static str {
            "ok"
        }
        // Static segments beside a parameter: axum refuses a conflict by
        // panicking here, not when a request comes.
        let _ = axum::Router::<()>::new()
            .route("/burns", axum::routing::get(ok))
            .route("/burns/days", axum::routing::get(ok))
            .route("/burns/top", axum::routing::get(ok))
            .route("/burns/recent", axum::routing::get(ok))
            .route("/burns/payload", axum::routing::get(ok))
            .route("/burns/notary/:hash", axum::routing::get(ok))
            .route("/burns/:purpose/:subject", axum::routing::get(ok));
    }

    #[test]
    fn totals_read_by_kind_and_purpose() {
        let mut t = Totals::default();
        t.names = 5;
        t.backing[Purpose::POOL.0 as usize] = 7;
        let v = totals_json(&t);
        assert_eq!((v["burned"].as_u64(), v["backing"].as_u64(), v["by_purpose"]["pool"].as_u64()), (Some(12), Some(7), Some(7)));
        assert_eq!(share(1, 4), json!(0.25));
        assert_eq!(share(1, 0), Value::Null);
    }
}
