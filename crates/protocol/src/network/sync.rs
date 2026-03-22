//! Time synchronization messages
//!
//! Protocol for synchronizing clocks across devices using a simplified PTP-like approach.

use serde::{Deserialize, Serialize};

/// Sync message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncMessage {
    /// Time sync request (like PTP Sync)
    TimeRequest(TimeSyncRequest),

    /// Time sync response (like PTP Follow_Up + Delay_Resp)
    TimeResponse(TimeSyncResponse),

    /// Clock adjustment notification
    ClockAdjustment {
        /// Offset to add to local clock (microseconds)
        offset_us: i64,
        /// Estimated one-way latency (microseconds)
        latency_us: u64,
    },

    /// Request current execution clock state
    QueryExecutionClock,

    /// Response with execution clock state
    ExecutionClockState {
        /// Current bar
        bar: Option<u32>,
        /// Current beat
        beat: Option<u32>,
        /// Beat phase (0.0 - 1.0)
        beat_phase: Option<f32>,
        /// Tempo (BPM)
        tempo_bpm: Option<f32>,
        /// Coupling mode
        coupling_mode: crate::CouplingMode,
        /// Timestamp when this state was captured
        timestamp_us: i64,
    },

    /// Metronome tick (for audible click sync)
    MetronomeTick {
        /// Beat number
        beat: u32,
        /// Timestamp when tick should occur
        timestamp_us: i64,
    },
}

/// Time synchronization request
///
/// Similar to PTP Sync message. Initiator sends this to peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeSyncRequest {
    /// Sequence ID (for matching request/response)
    pub sequence_id: u32,

    /// Timestamp when request was sent (initiator's clock, microseconds)
    pub t1_send_us: i64,

    /// Request initiator's device ID
    pub initiator_id: String,
}

impl TimeSyncRequest {
    /// Create a new time sync request
    pub fn new(initiator_id: impl Into<String>) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQUENCE: AtomicU32 = AtomicU32::new(1);

        Self {
            sequence_id: SEQUENCE.fetch_add(1, Ordering::Relaxed),
            t1_send_us: Self::now_us(),
            initiator_id: initiator_id.into(),
        }
    }

    fn now_us() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}

/// Time synchronization response
///
/// Peer responds with timestamps to compute offset and latency.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeSyncResponse {
    /// Sequence ID (matches request)
    pub sequence_id: u32,

    /// Timestamp from original request (t1)
    pub t1_send_us: i64,

    /// Timestamp when peer received request (peer's clock, microseconds)
    pub t2_receive_us: i64,

    /// Timestamp when peer sent response (peer's clock, microseconds)
    pub t3_send_us: i64,

    /// Peer's device ID
    pub peer_id: String,
}

impl TimeSyncResponse {
    /// Create a response to a request
    pub fn from_request(request: &TimeSyncRequest, peer_id: impl Into<String>) -> Self {
        let now = Self::now_us();
        Self {
            sequence_id: request.sequence_id,
            t1_send_us: request.t1_send_us,
            t2_receive_us: now,
            t3_send_us: now, // Same for immediate response
            peer_id: peer_id.into(),
        }
    }

    /// Compute clock offset and round-trip time
    ///
    /// Returns (offset_us, rtt_us) where:
    /// - offset_us: How much to add to local clock to match peer
    /// - rtt_us: Round-trip time
    ///
    /// Requires t4_receive_us: timestamp when initiator received this response
    pub fn compute_offset(&self, t4_receive_us: i64) -> (i64, u64) {
        // Classic NTP/PTP offset calculation:
        // offset = ((t2 - t1) + (t3 - t4)) / 2
        // rtt = (t4 - t1) - (t3 - t2)

        let t1 = self.t1_send_us;
        let t2 = self.t2_receive_us;
        let t3 = self.t3_send_us;
        let t4 = t4_receive_us;

        let offset = ((t2 - t1) + (t3 - t4)) / 2;
        let rtt = ((t4 - t1) - (t3 - t2)).max(0) as u64;

        (offset, rtt)
    }

    fn now_us() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}

/// Time synchronization statistics
///
/// Track sync quality over multiple exchanges.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyncStats {
    /// Number of sync exchanges completed
    pub exchanges: u32,

    /// Average offset (microseconds)
    pub avg_offset_us: f64,

    /// Offset standard deviation (microseconds)
    pub offset_stddev_us: f64,

    /// Average round-trip time (microseconds)
    pub avg_rtt_us: f64,

    /// Minimum RTT observed (microseconds)
    pub min_rtt_us: u64,

    /// Maximum RTT observed (microseconds)
    pub max_rtt_us: u64,

    /// Last sync timestamp (microseconds)
    pub last_sync_us: i64,

    /// Sync quality (0.0 - 1.0)
    ///
    /// Based on RTT stability and offset consistency.
    /// 1.0 = excellent sync, 0.0 = poor sync
    pub quality: f32,
}

impl SyncStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            exchanges: 0,
            avg_offset_us: 0.0,
            offset_stddev_us: 0.0,
            avg_rtt_us: 0.0,
            min_rtt_us: u64::MAX,
            max_rtt_us: 0,
            last_sync_us: 0,
            quality: 0.0,
        }
    }

    /// Update stats with new sync result
    pub fn update(&mut self, offset_us: i64, rtt_us: u64, timestamp_us: i64) {
        let n = self.exchanges as f64;
        let new_n = (self.exchanges + 1) as f64;

        // Update averages
        self.avg_offset_us = (self.avg_offset_us * n + offset_us as f64) / new_n;
        self.avg_rtt_us = (self.avg_rtt_us * n + rtt_us as f64) / new_n;

        // Update min/max RTT
        self.min_rtt_us = self.min_rtt_us.min(rtt_us);
        self.max_rtt_us = self.max_rtt_us.max(rtt_us);

        self.exchanges += 1;
        self.last_sync_us = timestamp_us;

        // Compute quality based on RTT stability
        // Quality is high when RTT is low and stable
        let rtt_range = (self.max_rtt_us - self.min_rtt_us) as f64;
        let rtt_stability = 1.0 - (rtt_range / self.avg_rtt_us.max(1.0)).min(1.0);
        let rtt_quality = 1.0 - (self.avg_rtt_us / 10000.0).min(1.0); // 10ms = poor

        self.quality = ((rtt_stability * 0.6 + rtt_quality * 0.4) as f32).clamp(0.0, 1.0);
    }

    /// Check if sync is good (quality > 0.7)
    pub fn is_good(&self) -> bool {
        self.quality > 0.7 && self.exchanges >= 3
    }

    /// Check if sync is excellent (quality > 0.9)
    pub fn is_excellent(&self) -> bool {
        self.quality > 0.9 && self.exchanges >= 5
    }

    /// Get age since last sync (seconds)
    pub fn age_seconds(&self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        (now - self.last_sync_us) as f64 / 1_000_000.0
    }
}

impl Default for SyncStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_sync_request() {
        let req = TimeSyncRequest::new("device1");
        assert_eq!(req.initiator_id, "device1");
        assert!(req.t1_send_us > 0);
    }

    #[test]
    fn test_time_sync_response() {
        let req = TimeSyncRequest::new("device1");
        let resp = TimeSyncResponse::from_request(&req, "device2");

        assert_eq!(resp.sequence_id, req.sequence_id);
        assert_eq!(resp.t1_send_us, req.t1_send_us);
        assert_eq!(resp.peer_id, "device2");
    }

    #[test]
    fn test_offset_computation() {
        // Simulate perfect sync (no offset, instant transmission)
        let t1 = 1000;
        let t2 = 1000;
        let t3 = 1001;
        let t4 = 1001;

        let resp = TimeSyncResponse {
            sequence_id: 1,
            t1_send_us: t1,
            t2_receive_us: t2,
            t3_send_us: t3,
            peer_id: "peer".to_string(),
        };

        let (offset, rtt) = resp.compute_offset(t4);
        assert_eq!(offset, 0); // Perfect sync
        assert_eq!(rtt, 0); // Instant
    }

    #[test]
    fn test_offset_with_delay() {
        // Initiator at t=1000, peer at t=2000 (1000us ahead)
        // 100us network delay each way
        let t1 = 1000; // Initiator sends
        let t2 = 2100; // Peer receives (peer clock 1000 ahead + 100 delay)
        let t3 = 2101; // Peer responds
        let t4 = 1201; // Initiator receives (100 delay back)

        let resp = TimeSyncResponse {
            sequence_id: 1,
            t1_send_us: t1,
            t2_receive_us: t2,
            t3_send_us: t3,
            peer_id: "peer".to_string(),
        };

        let (offset, rtt) = resp.compute_offset(t4);
        assert!(offset > 900 && offset < 1100); // ~1000us offset
        assert!(rtt > 150 && rtt < 250); // ~200us RTT
    }

    #[test]
    fn test_sync_stats_update() {
        let mut stats = SyncStats::new();
        assert_eq!(stats.exchanges, 0);

        stats.update(100, 1000, 1000000);
        assert_eq!(stats.exchanges, 1);
        assert_eq!(stats.avg_offset_us, 100.0);
        assert_eq!(stats.min_rtt_us, 1000);

        stats.update(110, 1100, 2000000);
        assert_eq!(stats.exchanges, 2);
        assert_eq!(stats.avg_offset_us, 105.0);
        assert_eq!(stats.min_rtt_us, 1000);
        assert_eq!(stats.max_rtt_us, 1100);
    }

    #[test]
    fn test_sync_quality() {
        let mut stats = SyncStats::new();

        // Add several consistent measurements
        for _ in 0..5 {
            stats.update(100, 1000, 1000000);
        }

        assert!(stats.is_good());
        assert!(stats.quality > 0.7);
    }

    #[test]
    fn test_metronome_tick() {
        let tick = SyncMessage::MetronomeTick {
            beat: 4,
            timestamp_us: 1000000,
        };

        match tick {
            SyncMessage::MetronomeTick { beat, .. } => {
                assert_eq!(beat, 4);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_serialization() {
        let req = TimeSyncRequest::new("device1");
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TimeSyncRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.sequence_id, deserialized.sequence_id);
    }
}
