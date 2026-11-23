//! Stamps, worked off the wallet's back.
//!
//! A sealed message to someone who has not written back carries a stamp
//! of the bits the recipient's name asks (its `postage` record): a second
//! of work at the standard price, hours if the name asks the most. The
//! wallet's actor cannot wait on that, so one worker thread does the work,
//! on every core, and says when a stamp is done.
//!
//! It always works the cheapest stamp waiting, then the oldest: a message
//! to a name that asks a lot holds up nothing else, and a cheaper one
//! coming in stops the dear one until it is done. Stopping costs nothing.
//! A stamp is found by trying nonces from a random start, and each try is
//! as likely to succeed as any other, so work taken up again, after
//! another stamp or at another price, is as near to done as the work that
//! was set aside.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// What the worker says of a stamp it finished: the job, the bits it was
/// worked to, and the nonce that shows them.
pub type Done = Box<dyn Fn(u64, u32, u64) + Send + 'static>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Job {
    id: u64,
    key: [u8; 32],
    bits: u32,
}

#[derive(Default)]
struct Queue {
    jobs: Vec<Job>,
    /// The job being worked, the bits it is worked to, and what stops it.
    running: Option<(u64, u32, Arc<AtomicBool>)>,
    closed: bool,
}

impl Queue {
    /// The job to work next: the cheapest, then the oldest.
    fn next(&self) -> Option<Job> {
        self.jobs.iter().min_by_key(|j| (j.bits, j.id)).copied()
    }

    /// Stops the running job if it is `id`, or dearer than `bits`.
    fn stop_for(&self, id: u64, bits: u32) {
        if let Some((running, at, stop)) = &self.running {
            if *running == id || *at > bits {
                stop.store(true, Ordering::Release);
            }
        }
    }
}

struct Shared {
    q: Mutex<Queue>,
    cv: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The stamp worker. Dropping it stops the work and the thread.
pub struct Stamper {
    shared: Arc<Shared>,
}

impl Stamper {
    /// A worker that calls `done` (on its own thread) as each stamp is done.
    pub fn new(done: Done) -> Self {
        let shared = Arc::new(Shared { q: Mutex::new(Queue::default()), cv: Condvar::new() });
        let s = shared.clone();
        std::thread::Builder::new().name("sealed stamps".into()).spawn(move || work(&s, &done)).expect("a thread for stamps");
        Stamper { shared }
    }

    /// Work a stamp of `bits` under `key`, as job `id`. A job already
    /// waiting under `id` is replaced.
    pub fn submit(&self, id: u64, key: [u8; 32], bits: u32) {
        let mut q = self.shared.lock();
        q.jobs.retain(|j| j.id != id);
        q.jobs.push(Job { id, key, bits });
        q.stop_for(id, bits);
        drop(q);
        self.shared.cv.notify_all();
    }

    /// Job `id` is now worth `bits`: higher, and the work goes on to it;
    /// lower, and what was done may already be enough.
    pub fn reprice(&self, id: u64, bits: u32) {
        let mut q = self.shared.lock();
        let Some(j) = q.jobs.iter_mut().find(|j| j.id == id) else { return };
        if j.bits == bits {
            return;
        }
        j.bits = bits;
        q.stop_for(id, bits);
        drop(q);
        self.shared.cv.notify_all();
    }

    /// Job `id` is not wanted: it is dropped, and stopped if it is running.
    pub fn cancel(&self, id: u64) {
        let mut q = self.shared.lock();
        q.jobs.retain(|j| j.id != id);
        if let Some((running, _, stop)) = &q.running {
            if *running == id {
                stop.store(true, Ordering::Release);
            }
        }
    }

    /// Every job is dropped.
    pub fn clear(&self) {
        let mut q = self.shared.lock();
        q.jobs.clear();
        if let Some((_, _, stop)) = &q.running {
            stop.store(true, Ordering::Release);
        }
    }

    /// The bits job `id` is worked to, if it is waiting or running.
    pub fn bits(&self, id: u64) -> Option<u32> {
        self.shared.lock().jobs.iter().find(|j| j.id == id).map(|j| j.bits)
    }

    /// The job being worked now.
    pub fn running(&self) -> Option<u64> {
        self.shared.lock().running.as_ref().map(|(id, ..)| *id)
    }
}

impl Drop for Stamper {
    fn drop(&mut self) {
        let mut q = self.shared.lock();
        q.closed = true;
        q.jobs.clear();
        if let Some((_, _, stop)) = &q.running {
            stop.store(true, Ordering::Release);
        }
        drop(q);
        self.shared.cv.notify_all();
    }
}

fn work(shared: &Shared, done: &Done) {
    loop {
        let (job, stop) = {
            let mut q = shared.lock();
            let job = loop {
                if q.closed {
                    return;
                }
                if let Some(j) = q.next() {
                    break j;
                }
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            };
            let stop = Arc::new(AtomicBool::new(false));
            q.running = Some((job.id, job.bits, stop.clone()));
            (job, stop)
        };
        let found = midstate::sealed::mine_stamp_until(&job.key, job.bits, &stop);
        let mut q = shared.lock();
        q.running = None;
        let Some(nonce) = found else { continue };
        // Done if it is still wanted, at no more than it was worked to: a
        // job made dearer meanwhile goes round again.
        let Some(i) = q.jobs.iter().position(|j| j.id == job.id && j.key == job.key && j.bits <= job.bits) else { continue };
        q.jobs.remove(i);
        drop(q);
        done(job.id, job.bits, nonce);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::sealed::stamp_bits;
    use std::sync::mpsc;
    use std::time::Duration;

    fn stamper() -> (Stamper, mpsc::Receiver<(u64, u32, u64)>) {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        (Stamper::new(Box::new(move |id, bits, nonce| {
            let _ = tx.lock().unwrap().send((id, bits, nonce));
        })), rx)
    }

    fn next(rx: &mpsc::Receiver<(u64, u32, u64)>) -> (u64, u32, u64) {
        rx.recv_timeout(Duration::from_secs(60)).expect("a stamp")
    }

    /// Waits until job `id` is the one being worked.
    fn until_running(s: &Stamper, id: u64) {
        for _ in 0..2_000 {
            if s.running() == Some(id) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("job {id} never ran");
    }

    #[test]
    fn the_cheapest_stamp_is_worked_first_then_the_oldest() {
        let job = |id, bits| Job { id, key: [id as u8; 32], bits };
        let mut q = Queue::default();
        assert_eq!(q.next(), None);
        q.jobs = vec![job(1, 30), job(2, 24), job(3, 24), job(4, 28)];
        assert_eq!(q.next().map(|j| j.id), Some(2));
        q.jobs.remove(1);
        assert_eq!(q.next().map(|j| j.id), Some(3));
    }

    #[test]
    fn a_dear_stamp_holds_up_nothing() {
        let (s, rx) = stamper();
        // Far too dear to finish here: it runs until something cheaper comes.
        s.submit(1, [1; 32], 62);
        until_running(&s, 1);
        s.submit(2, [2; 32], 10);
        s.submit(3, [3; 32], 8);
        let (a, b) = (next(&rx), next(&rx));
        let mut ids = [a.0, b.0];
        ids.sort();
        assert_eq!(ids, [2, 3], "both go ahead of the dear one");
        for (id, bits, nonce) in [a, b] {
            assert!(stamp_bits(&[id as u8; 32], nonce) >= bits);
        }
        // The dear one is back at work, and goes once it is cheap enough.
        until_running(&s, 1);
        assert_eq!(s.bits(1), Some(62));
        s.reprice(1, 9);
        let c = next(&rx);
        assert_eq!((c.0, c.1), (1, 9));
        assert!(stamp_bits(&[1; 32], c.2) >= 9);
        assert_eq!(s.bits(1), None, "done jobs leave");
    }

    #[test]
    fn a_cancelled_stamp_never_comes_and_a_replaced_one_comes_once() {
        let (s, rx) = stamper();
        s.submit(1, [1; 32], 62);
        until_running(&s, 1);
        s.cancel(1);
        assert_eq!(s.bits(1), None);
        // Replaced under the same id: the new key's stamp comes, once.
        s.submit(3, [3; 32], 62);
        s.reprice(3, 63);
        assert_eq!(s.bits(3), Some(63));
        s.submit(3, [4; 32], 7);
        let (id, bits, nonce) = next(&rx);
        assert_eq!((id, bits), (3, 7));
        assert!(stamp_bits(&[4; 32], nonce) >= 7);
        s.clear();
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err(), "nothing else comes");
    }
}
