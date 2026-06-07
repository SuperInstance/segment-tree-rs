//! Segment tree supporting point updates, range queries, and lazy propagation.
//!
//! # Examples
//!
//! ```
//! use segment_tree::{SegTree, LazySegTree};
//!
//! let mut st = SegTree::new_sum(5);
//! st.update(2, 10);
//! assert_eq!(st.query(0, 5), 10);
//!
//! let mut lazy = LazySegTree::from_slice(&[1, 2, 3, 4, 5]);
//! lazy.range_add(1, 3, 10); // add 10 to indices 1..=3
//! assert_eq!(lazy.range_sum(0, 4), 45);
//! ```

use std::cmp;

// ── helpers ──────────────────────────────────────────────────────────────────

pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        a %= b;
        std::mem::swap(&mut a, &mut b);
    }
    a
}

// ── SegTree ───────────────────────────────────────────────────────────────────

/// Iterative segment tree (bottom-up) for point updates and range queries.
///
/// The tree stores values at leaf positions `[0, n)` and answers
/// associative range queries in O(log n).
pub struct SegTree {
    n: usize,
    tree: Vec<i64>,
    identity: i64,
    op: fn(i64, i64) -> i64,
}

impl SegTree {
    /// Build an empty segment tree of size `n` with the given binary operation
    /// and identity element.
    pub fn new(n: usize, op: fn(i64, i64) -> i64, identity: i64) -> Self {
        // Use the next power-of-two to keep indexing simple.
        let size = n.next_power_of_two();
        Self { n: size, tree: vec![identity; 2 * size], identity, op }
    }

    /// Build from a slice.
    pub fn from_slice(data: &[i64], op: fn(i64, i64) -> i64, identity: i64) -> Self {
        let size = data.len().next_power_of_two();
        let mut tree = vec![identity; 2 * size];
        for (i, &v) in data.iter().enumerate() {
            tree[size + i] = v;
        }
        for i in (1..size).rev() {
            tree[i] = op(tree[2 * i], tree[2 * i + 1]);
        }
        Self { n: size, tree, identity, op }
    }

    /// Convenience: sum segment tree.
    pub fn new_sum(n: usize) -> Self {
        Self::new(n, |a, b| a + b, 0)
    }

    /// Convenience: min segment tree.
    pub fn new_min(n: usize) -> Self {
        Self::new(n, cmp::min, i64::MAX)
    }

    /// Convenience: max segment tree.
    pub fn new_max(n: usize) -> Self {
        Self::new(n, cmp::max, i64::MIN)
    }

    /// Convenience: gcd segment tree.
    pub fn new_gcd(n: usize) -> Self {
        Self::new(n, gcd, 0)
    }

    /// Set `tree[pos] = val` (0-indexed) and update ancestors.
    pub fn update(&mut self, pos: usize, val: i64) {
        let mut i = self.n + pos;
        self.tree[i] = val;
        i >>= 1;
        while i >= 1 {
            self.tree[i] = (self.op)(self.tree[2 * i], self.tree[2 * i + 1]);
            if i == 1 {
                break;
            }
            i >>= 1;
        }
    }

    /// Query the range `[l, r)` (exclusive right).
    pub fn query(&self, mut l: usize, mut r: usize) -> i64 {
        let mut left_acc = self.identity;
        let mut right_acc = self.identity;
        l += self.n;
        r += self.n;
        while l < r {
            if l & 1 == 1 {
                left_acc = (self.op)(left_acc, self.tree[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                right_acc = (self.op)(self.tree[r], right_acc);
            }
            l >>= 1;
            r >>= 1;
        }
        (self.op)(left_acc, right_acc)
    }

    /// Query a single element.
    #[inline]
    pub fn get(&self, pos: usize) -> i64 {
        self.tree[self.n + pos]
    }
}

// ── LazySegTree ───────────────────────────────────────────────────────────────

/// Recursive segment tree with lazy propagation.
///
/// Supports:
/// - `range_add(l, r, delta)` — add `delta` to all elements in `[l, r]` (inclusive)
/// - `range_sum(l, r)` — sum of all elements in `[l, r]` (inclusive)
/// - `range_min(l, r)` — minimum in `[l, r]`
/// - `range_max(l, r)` — maximum in `[l, r]`
pub struct LazySegTree {
    n: usize,
    sum: Vec<i64>,
    min: Vec<i64>,
    max: Vec<i64>,
    lazy: Vec<i64>,
}

impl LazySegTree {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            sum: vec![0; 4 * n],
            min: vec![0; 4 * n],
            max: vec![0; 4 * n],
            lazy: vec![0; 4 * n],
        }
    }

    pub fn from_slice(data: &[i64]) -> Self {
        let n = data.len();
        let mut st = Self::new(n);
        if n > 0 {
            st.build(data, 1, 0, n - 1);
        }
        st
    }

    fn build(&mut self, data: &[i64], node: usize, start: usize, end: usize) {
        if start == end {
            self.sum[node] = data[start];
            self.min[node] = data[start];
            self.max[node] = data[start];
        } else {
            let mid = (start + end) / 2;
            self.build(data, 2 * node, start, mid);
            self.build(data, 2 * node + 1, mid + 1, end);
            self.pull_up(node);
        }
    }

    #[inline]
    fn pull_up(&mut self, node: usize) {
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1];
        self.min[node] = cmp::min(self.min[2 * node], self.min[2 * node + 1]);
        self.max[node] = cmp::max(self.max[2 * node], self.max[2 * node + 1]);
    }

    fn push_down(&mut self, node: usize, start: usize, end: usize) {
        if self.lazy[node] != 0 {
            let mid = (start + end) / 2;
            let lc = 2 * node;
            let rc = 2 * node + 1;
            let left_len = (mid - start + 1) as i64;
            let right_len = (end - mid) as i64;
            let delta = self.lazy[node];

            self.sum[lc] += delta * left_len;
            self.min[lc] += delta;
            self.max[lc] += delta;
            self.lazy[lc] += delta;

            self.sum[rc] += delta * right_len;
            self.min[rc] += delta;
            self.max[rc] += delta;
            self.lazy[rc] += delta;

            self.lazy[node] = 0;
        }
    }

    /// Add `delta` to every element in `[l, r]` (inclusive, 0-indexed).
    pub fn range_add(&mut self, l: usize, r: usize, delta: i64) {
        let n = self.n;
        self.do_add(1, 0, n - 1, l, r, delta);
    }

    fn do_add(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize, delta: i64) {
        if r < start || end < l {
            return;
        }
        if l <= start && end <= r {
            let len = (end - start + 1) as i64;
            self.sum[node] += delta * len;
            self.min[node] += delta;
            self.max[node] += delta;
            self.lazy[node] += delta;
            return;
        }
        self.push_down(node, start, end);
        let mid = (start + end) / 2;
        self.do_add(2 * node, start, mid, l, r, delta);
        self.do_add(2 * node + 1, mid + 1, end, l, r, delta);
        self.pull_up(node);
    }

    /// Sum of elements in `[l, r]` (inclusive, 0-indexed).
    pub fn range_sum(&mut self, l: usize, r: usize) -> i64 {
        let n = self.n;
        self.do_sum(1, 0, n - 1, l, r)
    }

    fn do_sum(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize) -> i64 {
        if r < start || end < l {
            return 0;
        }
        if l <= start && end <= r {
            return self.sum[node];
        }
        self.push_down(node, start, end);
        let mid = (start + end) / 2;
        self.do_sum(2 * node, start, mid, l, r) + self.do_sum(2 * node + 1, mid + 1, end, l, r)
    }

    /// Minimum in `[l, r]` (inclusive, 0-indexed).
    pub fn range_min(&mut self, l: usize, r: usize) -> i64 {
        let n = self.n;
        self.do_min(1, 0, n - 1, l, r)
    }

    fn do_min(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize) -> i64 {
        if r < start || end < l {
            return i64::MAX;
        }
        if l <= start && end <= r {
            return self.min[node];
        }
        self.push_down(node, start, end);
        let mid = (start + end) / 2;
        cmp::min(
            self.do_min(2 * node, start, mid, l, r),
            self.do_min(2 * node + 1, mid + 1, end, l, r),
        )
    }

    /// Maximum in `[l, r]` (inclusive, 0-indexed).
    pub fn range_max(&mut self, l: usize, r: usize) -> i64 {
        let n = self.n;
        self.do_max(1, 0, n - 1, l, r)
    }

    fn do_max(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize) -> i64 {
        if r < start || end < l {
            return i64::MIN;
        }
        if l <= start && end <= r {
            return self.max[node];
        }
        self.push_down(node, start, end);
        let mid = (start + end) / 2;
        cmp::max(
            self.do_max(2 * node, start, mid, l, r),
            self.do_max(2 * node + 1, mid + 1, end, l, r),
        )
    }

    /// Point update: set element at position `pos` to `val`.
    pub fn set(&mut self, pos: usize, val: i64) {
        let n = self.n;
        self.do_set(1, 0, n - 1, pos, val);
    }

    fn do_set(&mut self, node: usize, start: usize, end: usize, pos: usize, val: i64) {
        if start == end {
            self.sum[node] = val;
            self.min[node] = val;
            self.max[node] = val;
            self.lazy[node] = 0;
            return;
        }
        self.push_down(node, start, end);
        let mid = (start + end) / 2;
        if pos <= mid {
            self.do_set(2 * node, start, mid, pos, val);
        } else {
            self.do_set(2 * node + 1, mid + 1, end, pos, val);
        }
        self.pull_up(node);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SegTree sum ──────────────────────────────────────────────────────────

    #[test]
    fn sum_empty_build() {
        let st = SegTree::new_sum(8);
        assert_eq!(st.query(0, 8), 0);
    }

    #[test]
    fn sum_point_updates() {
        let mut st = SegTree::new_sum(8);
        st.update(0, 5);
        st.update(3, 7);
        st.update(7, 2);
        assert_eq!(st.query(0, 8), 14);
        assert_eq!(st.query(0, 4), 12);
        assert_eq!(st.query(4, 8), 2);
    }

    #[test]
    fn sum_from_slice() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        let st = SegTree::from_slice(&data, |a, b| a + b, 0);
        assert_eq!(st.query(0, 8), 36);
        assert_eq!(st.query(2, 5), 12);
    }

    #[test]
    fn sum_overwrite() {
        let mut st = SegTree::new_sum(4);
        st.update(1, 10);
        assert_eq!(st.query(0, 4), 10);
        st.update(1, 3);
        assert_eq!(st.query(0, 4), 3);
    }

    // ── SegTree min/max ──────────────────────────────────────────────────────

    #[test]
    fn min_query() {
        let data = [5, 3, 8, 1, 9, 2, 7, 4];
        let st = SegTree::from_slice(&data, cmp::min, i64::MAX);
        assert_eq!(st.query(0, 8), 1);
        assert_eq!(st.query(0, 3), 3);
        assert_eq!(st.query(5, 8), 2);
    }

    #[test]
    fn max_query() {
        let data = [5, 3, 8, 1, 9, 2, 7, 4];
        let st = SegTree::from_slice(&data, cmp::max, i64::MIN);
        assert_eq!(st.query(0, 8), 9);
        assert_eq!(st.query(0, 3), 8);
        assert_eq!(st.query(3, 6), 9);
    }

    #[test]
    fn min_after_update() {
        let mut st = SegTree::new_min(5);
        for i in 0..5 {
            st.update(i, (5 - i) as i64);
        }
        assert_eq!(st.query(0, 5), 1);
        st.update(4, 10);
        assert_eq!(st.query(0, 5), 2);
    }

    // ── SegTree gcd ──────────────────────────────────────────────────────────

    #[test]
    fn gcd_query() {
        let data = [12, 8, 6, 4];
        let st = SegTree::from_slice(&data, gcd, 0);
        assert_eq!(st.query(0, 4), 2);
        assert_eq!(st.query(0, 2), 4);
    }

    #[test]
    fn gcd_single() {
        let data = [42];
        let st = SegTree::from_slice(&data, gcd, 0);
        assert_eq!(st.query(0, 1), 42);
    }

    #[test]
    fn gcd_coprime() {
        let data = [6, 35];
        let st = SegTree::from_slice(&data, gcd, 0);
        assert_eq!(st.query(0, 2), 1);
    }

    // ── gcd helper ───────────────────────────────────────────────────────────

    #[test]
    fn gcd_helper() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(100, 75), 25);
    }

    // ── LazySegTree ──────────────────────────────────────────────────────────

    #[test]
    fn lazy_build_and_sum() {
        let mut st = LazySegTree::from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(st.range_sum(0, 4), 15);
        assert_eq!(st.range_sum(1, 3), 9);
    }

    #[test]
    fn lazy_range_add_sum() {
        let mut st = LazySegTree::from_slice(&[1, 2, 3, 4, 5]);
        st.range_add(1, 3, 10);
        assert_eq!(st.range_sum(0, 4), 45);
        assert_eq!(st.range_sum(1, 3), 39);
        assert_eq!(st.range_sum(0, 0), 1);
        assert_eq!(st.range_sum(4, 4), 5);
    }

    #[test]
    fn lazy_range_add_min_max() {
        let mut st = LazySegTree::from_slice(&[5, 3, 8, 1, 9]);
        assert_eq!(st.range_min(0, 4), 1);
        assert_eq!(st.range_max(0, 4), 9);
        st.range_add(0, 4, 100);
        assert_eq!(st.range_min(0, 4), 101);
        assert_eq!(st.range_max(0, 4), 109);
    }

    #[test]
    fn lazy_range_add_partial() {
        let mut st = LazySegTree::from_slice(&[0, 0, 0, 0, 0]);
        st.range_add(2, 4, 5);
        assert_eq!(st.range_sum(0, 4), 15);
        assert_eq!(st.range_sum(0, 1), 0);
        assert_eq!(st.range_min(0, 1), 0);
        assert_eq!(st.range_min(2, 4), 5);
    }

    #[test]
    fn lazy_set_point() {
        let mut st = LazySegTree::from_slice(&[10, 20, 30]);
        st.set(1, 5);
        assert_eq!(st.range_sum(0, 2), 45);
        assert_eq!(st.range_min(0, 2), 5);
    }

    #[test]
    fn lazy_multiple_adds() {
        let mut st = LazySegTree::from_slice(&[0, 0, 0]);
        st.range_add(0, 2, 3);
        st.range_add(1, 2, 2);
        st.range_add(2, 2, 1);
        // [3, 5, 6]
        assert_eq!(st.range_sum(0, 2), 14);
        assert_eq!(st.range_min(0, 2), 3);
        assert_eq!(st.range_max(0, 2), 6);
    }

    #[test]
    fn lazy_single_element() {
        let mut st = LazySegTree::from_slice(&[42]);
        assert_eq!(st.range_sum(0, 0), 42);
        st.range_add(0, 0, 8);
        assert_eq!(st.range_sum(0, 0), 50);
        assert_eq!(st.range_min(0, 0), 50);
        assert_eq!(st.range_max(0, 0), 50);
    }

    #[test]
    fn lazy_negative_values() {
        let mut st = LazySegTree::from_slice(&[-5, -3, -1, 0, 2]);
        assert_eq!(st.range_min(0, 4), -5);
        assert_eq!(st.range_max(0, 4), 2);
        st.range_add(0, 4, 5);
        assert_eq!(st.range_min(0, 4), 0);
        assert_eq!(st.range_max(0, 4), 7);
        assert_eq!(st.range_sum(0, 4), 18);
    }

    #[test]
    fn segtree_get() {
        let data = [10, 20, 30];
        let st = SegTree::from_slice(&data, |a, b| a + b, 0);
        assert_eq!(st.get(0), 10);
        assert_eq!(st.get(1), 20);
        assert_eq!(st.get(2), 30);
    }
}
