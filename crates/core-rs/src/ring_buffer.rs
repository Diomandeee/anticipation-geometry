//! Lock-free ring buffer for control → audio thread communication.
//!
//! This implementation provides true lock-free SPSC (single producer,
//! single consumer) semantics using atomic operations.
//!
//! ## Design
//!
//! - Single producer (control thread)
//! - Single consumer (audio thread)
//! - Wait-free read/write
//! - Overwrite-on-full semantics (always keep the latest)
//!
//! ## Performance
//!
//! Unlike the Python implementation which uses threading.Lock, this
//! uses hardware atomics with appropriate memory ordering:
//! - `Release` on write (publish)
//! - `Acquire` on read (synchronize)
//!
//! This eliminates any possibility of lock contention and provides
//! predictable, bounded execution time suitable for real-time audio.

use portable_atomic::{AtomicU64, Ordering};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// Lock-free single-producer, single-consumer ring buffer.
///
/// # Semantics
///
/// - **Overwrite-on-full**: Writes never block; when full, the oldest
///   element is discarded and the new one is written.
/// - **read_latest()**: Returns the most recently written item or None.
/// - **available_count()**: Returns how many items are present.
///
/// # Thread Safety
///
/// This buffer is safe for exactly ONE producer thread and ONE consumer
/// thread. Using multiple producers or multiple consumers is undefined
/// behavior.
///
/// # Example
///
/// ```
/// use cc_core_rs::LockFreeRingBuffer;
///
/// let buf: LockFreeRingBuffer<f32> = LockFreeRingBuffer::new(4);
///
/// // Producer thread
/// buf.write(42.0);
/// buf.write(43.0);
///
/// // Consumer thread
/// if let Some(value) = buf.read_latest() {
///     assert_eq!(value, 43.0);
/// }
/// ```
pub struct LockFreeRingBuffer<T: Copy + Default> {
    /// Circular buffer storage
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Buffer capacity (power of 2 for efficient modulo)
    capacity: usize,
    /// Mask for fast modulo (capacity - 1)
    mask: usize,
    /// Monotonic write index (number of items ever written)
    write_idx: AtomicU64,
    /// Monotonic read index (number of items ever consumed)
    read_idx: AtomicU64,
}

// SAFETY: The buffer is designed for SPSC access patterns.
// The UnsafeCell contents are only accessed by one thread at a time
// due to the atomic index synchronization.
unsafe impl<T: Copy + Default + Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Copy + Default + Send> Sync for LockFreeRingBuffer<T> {}

impl<T: Copy + Default> LockFreeRingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// The capacity will be rounded up to the nearest power of 2
    /// for efficient index calculations.
    ///
    /// # Panics
    ///
    /// Panics if capacity is less than 2.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 2, "Capacity must be at least 2");

        // Round up to power of 2 for efficient modulo
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        // Allocate buffer with uninitialized slots
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask,
            write_idx: AtomicU64::new(0),
            read_idx: AtomicU64::new(0),
        }
    }

    /// Get the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // =========================================================================
    // Producer API
    // =========================================================================

    /// Write an item to the buffer (producer / control thread).
    ///
    /// # Behavior
    ///
    /// - Never blocks
    /// - If buffer is full, the oldest item is overwritten
    /// - Returns `true` if write succeeded without overwrite
    /// - Returns `false` if buffer was full and oldest entry was overwritten
    ///
    /// # Safety
    ///
    /// Must only be called from the single producer thread.
    #[inline]
    pub fn write(&self, item: T) -> bool {
        let write_idx = self.write_idx.load(Ordering::Relaxed);
        let read_idx = self.read_idx.load(Ordering::Acquire);

        let count = write_idx.wrapping_sub(read_idx);
        let was_full = count >= self.capacity as u64;

        // Calculate slot index using bitmask (fast modulo for power of 2)
        let slot = (write_idx as usize) & self.mask;

        // SAFETY: We're the only writer, and the slot at write_idx is
        // either unread (if not full) or will be skipped by advancing read_idx
        unsafe {
            (*self.buffer[slot].get()).write(item);
        }

        // Increment write index with Release ordering to publish the write
        self.write_idx.store(write_idx.wrapping_add(1), Ordering::Release);

        // If buffer was full, advance read index to discard oldest
        if was_full {
            self.read_idx.store(read_idx.wrapping_add(1), Ordering::Release);
        }

        !was_full
    }

    // =========================================================================
    // Consumer API
    // =========================================================================

    /// Read the most recent item (consumer / audio thread).
    ///
    /// # Behavior
    ///
    /// - Never blocks
    /// - Returns `None` if buffer is empty
    /// - Marks all items as consumed; subsequent calls will only see
    ///   items written after this call
    ///
    /// # Safety
    ///
    /// Must only be called from the single consumer thread.
    #[inline]
    pub fn read_latest(&self) -> Option<T> {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Relaxed);

        if write_idx == read_idx {
            // Buffer empty
            return None;
        }

        // Latest item is at write_idx - 1
        let latest_slot = ((write_idx.wrapping_sub(1)) as usize) & self.mask;

        // SAFETY: We're the only reader, and the slot is valid because
        // write_idx > read_idx, meaning there's at least one item
        let item = unsafe { (*self.buffer[latest_slot].get()).assume_init() };

        // Mark everything as consumed
        self.read_idx.store(write_idx, Ordering::Release);

        Some(item)
    }

    /// Peek at the most recent item without consuming it.
    ///
    /// # Behavior
    ///
    /// - Never blocks
    /// - Returns `None` if buffer is empty
    /// - Does not modify indices
    #[inline]
    pub fn peek_latest(&self) -> Option<T> {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Relaxed);

        if write_idx == read_idx {
            return None;
        }

        let latest_slot = ((write_idx.wrapping_sub(1)) as usize) & self.mask;

        // SAFETY: Same as read_latest
        let item = unsafe { (*self.buffer[latest_slot].get()).assume_init() };

        Some(item)
    }

    // =========================================================================
    // Introspection
    // =========================================================================

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Acquire);
        write_idx == read_idx
    }

    /// Check if buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Acquire);
        write_idx.wrapping_sub(read_idx) >= self.capacity as u64
    }

    /// Get number of unread items currently in the buffer.
    #[inline]
    pub fn available_count(&self) -> usize {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Acquire);

        let count = write_idx.wrapping_sub(read_idx);
        if count == 0 {
            return 0;
        }
        std::cmp::min(count as usize, self.capacity)
    }

    /// Clear the buffer.
    ///
    /// # Warning
    ///
    /// This is NOT safe to call concurrently with reads or writes.
    /// Call only when no other thread is using the buffer.
    pub fn clear(&self) {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        self.read_idx.store(write_idx, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_write_read() {
        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4);

        assert!(buf.is_empty());
        assert_eq!(buf.available_count(), 0);

        buf.write(42);
        assert!(!buf.is_empty());
        assert_eq!(buf.available_count(), 1);

        let value = buf.read_latest();
        assert_eq!(value, Some(42));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_overwrite_on_full() {
        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4);

        // Fill buffer
        for i in 0..4 {
            let not_full = buf.write(i);
            assert!(not_full);
        }

        assert!(buf.is_full());

        // Overwrite oldest
        let not_full = buf.write(100);
        assert!(!not_full); // Was full, overwrote

        // Should get the latest value
        assert_eq!(buf.read_latest(), Some(100));
    }

    #[test]
    fn test_peek_latest() {
        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4);

        buf.write(1);
        buf.write(2);
        buf.write(3);

        // Peek doesn't consume
        assert_eq!(buf.peek_latest(), Some(3));
        assert_eq!(buf.peek_latest(), Some(3));
        assert_eq!(buf.available_count(), 3);

        // Read consumes all
        assert_eq!(buf.read_latest(), Some(3));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_power_of_two_rounding() {
        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(5);
        assert_eq!(buf.capacity(), 8); // Rounded up to 8

        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4);
        assert_eq!(buf.capacity(), 4); // Already power of 2
    }

    #[test]
    fn test_clear() {
        let buf: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4);

        buf.write(1);
        buf.write(2);
        assert_eq!(buf.available_count(), 2);

        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.read_latest(), None);
    }
}
