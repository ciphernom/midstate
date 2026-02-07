use midstate::node::Node;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_corrupt_database_recovery() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db").join("state");
    
    // 1. Create a valid DB structure
    fs::create_dir_all(&db_path).unwrap();
    
    // 2. Write Garbage into the Sled data files
    fs::write(db_path.join("conf"), b"GARBAGE_DATA_CORRUPTION").unwrap();
    fs::write(db_path.join("db"), b"GARBAGE_DATA_CORRUPTION").unwrap();

    // 3. Try to open Node
    let result = Node::new(temp.path().into(), false, "127.0.0.1:0".parse().unwrap());

    // Depending on Sled's behavior, this should either return Err
    // or (if it auto-recovers) return Ok with genesis. 
    // We just want to ensure it doesn't Panic/Segfault.
    match result {
        Ok(node) => {
            // If it recovered, height should be 0 (Genesis)
            let (_handle, _) = node.create_handle();
            // We can't await in a non-async test, but we know new() is synchronous
            // If we are here, Sled recovered or ignored corruption.
        }
        Err(e) => {
            // This is also acceptable behavior for a corrupted DB
            println!("Node correctly failed to open corrupt DB: {}", e);
        }
    }
}
