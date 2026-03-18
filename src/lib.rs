//! This crate provides a container to group multiple, independent object allocations
//! together in memory. It does so by using buckets of uninitialized memory and tracking
//! the state for safe access.
//!
//! Each allocation is associated with an [`Id`]. The IDs don't have generation tracking
//! so, multiple different allocations can have identical IDs but their lifetimes will
//! never intersect.
//!
//! There are two main ways to allocate slots:
//!   - [`Pool::alloc`]
//!   - [`Pool::alloc_contiguous`]
//!
//! The former allocates a single element and consecutive calls to it have no guarantee
//! about how the element is allocated. The element may be allocated far away in an unused
//! slot of an item freed prior, or really close by.
//!
//! The latter takes an iterator of values and guarantees that all values are allocated
//! next to each other (and returns a slice to them).
//!
//! There are two main ways to query IDs:
//!   - [`Pool::get`] / [`Pool::get_mut`]
//!   - [`Pool::borrow_batch_mut`]
//!
//! The former options return a single (im)mutable reference to the specified item, or [`None`].
//! They also have unsafe counterparts ([`Pool::get_unchecked`] / [`Pool::get_mut_unchecked`]) which
//! have no runtime checks, similar to slices.
//!
//! The latter option offers a way to safely bypass Rust's borrow checker.
//! Similar to a slice, taking mutable references to two different elements is a completely valid
//! and safe thing to do, but the borrow checker cannot prove at compile time that this is safe.
//! Because of this very reason, the `.split_at` function was created.
//!
//! [`Pool::borrow_batch_mut`] offers to be the `.split_at` equivalent for [`Pool`]. Check its
//! documentation for more detail. Also see [`Pool::borrow_batch_mut_unchecked`] for a faster
//! but unsafe alternative.
//!
//! There are two main ways to deallocate slots and remove a value:
//!   - [`Pool::free`]
//!   - [`Pool::take`]
//!
//! The former simply drops the value and marks it as uninitialized. It's analogous to
//! calling [`drop`] on an [`Option`].
//!
//! The latter drops the value and returns it, if it exists. It's analogous to [`Option::take`].
//!
//! [`Pool`] also provides a way for the caller to manunally initialize an allocated slot
//! via [`Pool::alloc_uninit`]. An example use case would be to initialize a chonky structure
//! without copying memory around.
//!
//! [`Pool::alloc_uninit`] returns an [`Id`] and a mutable reference to a [`MaybeUninit`] for
//! manual initialization. The caller must ensure the slot is initialized before any read/writes.
//! See the [`Pool::alloc_uninit`] documentation for further information.
//!
//! ```
//! # use pool::Pool;
//! let mut p = Pool::new();
//!
//! let id_a = p.alloc(23);
//! p.free(id_a);
//!
//! let id_b = p.alloc(37);
//!
//! // With the current implementation, an ID which has been freed is immediately reused for
//! // single allocations.
//! assert_eq!(id_a, id_b);
//! ```

use std::{
    collections::HashSet,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use bit_vec::BitVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(u32, u32);

#[derive(Debug)]
struct Bucket<T> {
    items: Box<[MaybeUninit<T>]>,
    index: usize,
    init: BitVec,
}

impl<T> Deref for Bucket<T> {
    type Target = [MaybeUninit<T>];

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl<T> DerefMut for Bucket<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}

impl<T> Bucket<T> {
    fn new(capacity: usize) -> Self {
        Self {
            index: 0,
            init: BitVec::from_elem(capacity, false),
            items: (0..capacity)
                .map(|_| MaybeUninit::uninit())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn cap(&self) -> usize {
        self.items.len()
    }

    fn len(&self) -> usize {
        self.index
    }

    /// Marks an index as initialized and writes `init` into it.
    ///
    /// # Panics
    /// Panics if the specified index has previously been initialized.
    fn init(&mut self, index: usize, init: T) {
        if self.init[index] {
            panic!("double initialization");
        }

        self.init.set(index, true);
        self.items[index].write(init);
    }

    /// Mark a slot as initialized
    /// # Safety
    /// Initialization must be done by the caller.
    #[must_use = "initialization must be done by the caller"]
    unsafe fn mark_init(&mut self, index: usize) -> &mut MaybeUninit<T> {
        self.init.set(index, true);
        &mut self.items[index]
    }

    /// Checks whether the bucket has an item marked as uninitialized at its tail.
    fn has_trailing_space(&self) -> bool {
        self.index < self.cap()
    }

    /// Pushes an item at the end of the bucket and returns the index of the new item.
    ///
    /// # Panics
    /// Panics if the bucket runs out of trailing space.
    fn push(&mut self, init: T) -> usize {
        if self.index >= self.cap() {
            panic!("bucket overflowed");
        }

        self.init(self.index, init);
        self.index += 1;

        self.index - 1
    }

    /// Pushes an uninitialized item at the end of the bucket and marks it as initialized.
    /// # Safety
    /// Initialization must be done by the caller.
    #[must_use = "initialization must be done by the caller"]
    unsafe fn push_uninit(&mut self) -> (usize, &mut MaybeUninit<T>) {
        if self.index >= self.cap() {
            panic!("bucket overflowed");
        }

        self.init.set(self.index, true);

        self.index += 1;
        (self.index - 1, &mut self.items[self.index - 1])
    }

    /// Drops the value at the given index if it's marked as init.
    /// Returns whether a drop took place.
    fn drop_index(&mut self, index: usize) -> bool {
        if self.init.get(index).unwrap_or(false) {
            self.init.set(index, false);
            unsafe { self.items[index].assume_init_drop() };
            true
        } else {
            false
        }
    }

    /// Move the value at the given index out and mark it as uninitialized.
    fn take(&mut self, index: usize) -> Option<T> {
        if !self.init.get(index)? {
            return None;
        }

        self.init.set(index, false);
        unsafe { Some(self.items[index].assume_init_read()) }
    }

    fn trailing_space(&self) -> usize {
        self.cap() - self.index
    }

    fn get_init(&self, index: usize) -> Option<&T> {
        if !self.init.get(index)? {
            return None;
        }

        self.items
            .get(index)
            .map(|x| unsafe { x.assume_init_ref() })
    }

    fn get_init_mut(&mut self, index: usize) -> Option<&mut T> {
        if !self.init.get(index)? {
            return None;
        }

        self.items
            .get_mut(index)
            .map(|x| unsafe { x.assume_init_mut() })
    }
}

impl<T> Drop for Bucket<T> {
    fn drop(&mut self) {
        for (i, item) in self.items.iter_mut().enumerate() {
            if self.init[i] {
                unsafe { item.assume_init_drop() };
            }
        }
    }
}

#[derive(Debug)]
pub struct Pool<T> {
    buckets: Vec<Bucket<T>>,
    free_slots: Vec<Id>,
}

impl<T> Pool<T> {
    /// Creates an empty pool.
    pub fn new() -> Self {
        Self {
            buckets: vec![],
            free_slots: vec![],
        }
    }

    /// # Safety
    /// Initialization must be done by the caller
    unsafe fn mark_init(&mut self, slot: Id) -> &mut MaybeUninit<T> {
        let bucket = &mut self.buckets[slot.0 as usize];
        unsafe { bucket.mark_init(slot.1 as usize) }
    }

    fn new_bucket_size(&self) -> usize {
        // This number is kinda arbitrary.
        // Might be a good idea to scale it depending on the size of T
        const MAX_BUCKET_SIZE: usize = 65536;

        match self.buckets.last() {
            Some(bucket) => (bucket.cap() * 2).min(MAX_BUCKET_SIZE),
            None => 64,
        }
    }

    fn push_new_bucket(&mut self) -> &mut Bucket<T> {
        let bucket = Bucket::new(self.new_bucket_size());
        self.buckets.push(bucket);

        self.buckets.last_mut().unwrap()
    }

    /// Checks whether `id` points to an allocated, initialized value.
    pub fn is_valid(&self, id: Id) -> bool {
        self.get(id).is_some()
    }

    /// Allocates a new slot in the pool and initializes it.
    ///
    /// The allocation (in the following order) will:
    ///   - Check for free slots within already available buckets
    ///   - Allocate a new bucket
    /// until it finds a spot to put it.
    ///
    /// # Panics
    /// This function may panic if memory allocation fails.
    #[must_use = "discarding the id will make the allocated object inacessible"]
    pub fn alloc(&mut self, init: T) -> Id {
        let (id, x) = unsafe { self.alloc_uninit() };

        x.write(init);

        id
    }

    /// Allocates a new slot in the pool.
    ///
    /// The allocation follows the same logic as in [`Pool::alloc`].
    ///
    /// # Safety
    /// The allocated item is marked as initialized and it's the callers
    /// responsibility to ensure it's initialized (using the [`MaybeUninit`] handle)
    /// before access.
    pub unsafe fn alloc_uninit(&mut self) -> (Id, &mut MaybeUninit<T>) {
        if let Some(slot) = self.free_slots.pop() {
            return (slot, unsafe { self.mark_init(slot) });
        }

        let need_new_bucket = self
            .buckets
            .last()
            .map_or(true, |b| !b.has_trailing_space());

        if need_new_bucket {
            self.push_new_bucket();
        }

        let bucket_idx = self.buckets.len() as u32 - 1;
        let bucket = self.buckets.last_mut().unwrap();
        let (n, x) = unsafe { bucket.push_uninit() };

        let id = Id(bucket_idx, n as u32);
        (id, x)
    }

    pub fn get(&self, id: Id) -> Option<&T> {
        self.buckets.get(id.0 as usize)?.get_init(id.1 as usize)
    }

    /// # Safety
    /// `id` must be associated with an initialized item that is allocated AND initialized by this pool.
    pub unsafe fn get_unchecked(&self, id: Id) -> &T {
        unsafe { self.buckets[id.0 as usize][id.1 as usize].assume_init_ref() }
    }

    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.buckets
            .get_mut(id.0 as usize)?
            .get_init_mut(id.1 as usize)
    }

    /// # Safety
    /// `id` must be associated with an initialized item that is allocated AND initialized by this pool.
    pub unsafe fn get_mut_unchecked(&mut self, id: Id) -> &mut T {
        unsafe { self.buckets[id.0 as usize][id.1 as usize].assume_init_mut() }
    }

    /// Allocates multiple items with values provided from `sequence` next to each other.
    ///
    /// The new allocation does NOT reuse memory from previously freed allocations.
    ///
    /// The IDs of the allocated slots are pushed into `vec`. If you don't care about reusing
    /// a [`Vec`], see [`Pool::alloc_contiguous`].
    /// The return value is to the allocated slots.
    ///
    /// If `sequence` is empty, returns `&mut []` without doing anything.
    pub fn alloc_contiguous_into<S>(&mut self, sequence: S, vec: &mut Vec<Id>) -> &mut [T]
    where
        S: IntoIterator<Item = T, IntoIter: ExactSizeIterator>,
    {
        let it = sequence.into_iter();

        if it.len() == 0 {
            return &mut [];
        }

        let n_buckets = self.buckets.len() as u32;
        let bucket_idx;

        let bucket = match self.buckets.last_mut() {
            Some(bucket) if bucket.trailing_space() >= it.len() => {
                bucket_idx = n_buckets - 1;
                bucket
            }
            _ => {
                let size = self.new_bucket_size().max(it.len());
                bucket_idx = n_buckets;
                self.buckets.push(Bucket::new(size));
                self.buckets.last_mut().unwrap()
            }
        };

        let start = bucket.len();
        let mut end = 0;

        for value in it {
            end = bucket.push(value);
            vec.push(Id(bucket_idx as u32, end as u32));
        }

        let s = &mut bucket[start..=end];
        let n = s.len();

        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut T, n) }
    }

    /// Allocates multiple items with values provided from `sequence` next to each other.
    ///
    /// The new allocation does NOT reuse memory from previously freed allocations.
    ///
    /// If you want to reuse a [`Vec`] to store the IDs, see [`Pool::alloc_contiguous_into`].
    ///
    /// Returns the allocated slots' IDs and a reference to the memory block.
    /// If `sequence` is empty, returns `(vec![], &mut [])` without doing anything.
    #[must_use = "discarding the ids will make the allocated object inacessible"]
    pub fn alloc_contiguous<S>(&mut self, sequence: S) -> (Vec<Id>, &mut [T])
    where
        S: IntoIterator<Item = T, IntoIter: ExactSizeIterator>,
    {
        let it = sequence.into_iter();
        if it.len() == 0 {
            return (vec![], &mut []);
        }

        let mut v = Vec::with_capacity(it.len());

        let s = self.alloc_contiguous_into(it, &mut v);

        (v, s)
    }

    /// Borrow multiple elements mutably at once with different IDs.
    ///
    /// This function has no immutable counterpart as it's already possible to acive with [`Pool::get`].
    /// This function is the safe alternative to [`Pool::borrow_batch_mut_unchecked`].
    /// Internally, a [`HashSet`] is used in order to verify that no duplicate element is produced.
    ///
    /// # Panics
    /// This function is guaranteed to panic if `ids` produces an equivalent [`Id`] multiple times.
    ///
    /// ```should_panic
    /// # use pool::Pool;
    /// let mut p = Pool::new();
    /// let id = p.alloc(23);
    ///
    /// // This part is fine
    /// for r in p.borrow_batch_mut(std::iter::once(id)) {
    ///     *r.unwrap() = 37;
    /// }
    ///
    /// // This will panic!
    /// for r in p.borrow_batch_mut(std::iter::repeat_n(id, 4)) {
    ///     *r.unwrap() = 23;
    /// }
    /// ```
    #[must_use]
    pub fn borrow_batch_mut<S>(&mut self, ids: S) -> impl Iterator<Item = Option<&mut T>>
    where
        S: IntoIterator<Item = Id>,
    {
        let mut set = HashSet::new();
        let mut v = vec![];

        for id in ids.into_iter() {
            if set.contains(&id) {
                panic!("overlapping mutable borrow");
            }

            v.push(id);
            set.insert(id);
        }

        unsafe { self.borrow_batch_mut_unchecked(v) }
    }

    /// Borrow multiple elements mutably at once with different IDs.
    /// This function has no immutable counterpart as it's already possible to acive with [`Pool::get`].
    ///
    /// # Safety
    /// None of the [`Id`] values produced by the `ids` iterator should compare
    /// equal. If were to happen, it would break aliasing rules and be undefined
    /// behavior.
    ///
    /// ```rust,no_run
    /// # use pool::Pool;
    /// let mut p = Pool::new();
    ///
    /// let id = p.alloc(23);
    ///
    /// // Safe code, no aliasing violation
    /// let _ = unsafe { p.borrow_batch_mut_unchecked(std::iter::once(id)) };
    ///
    /// // !UNSAFE: multiple mutable references produced same element
    /// let _ = unsafe { p.borrow_batch_mut_unchecked(std::iter::repeat_n(id, 4)) };
    /// ```
    ///
    /// # Example
    /// ```
    /// # use pool::Pool;
    /// let mut p = Pool::new();
    ///
    /// let mut ids = vec![
    ///     p.alloc(0),
    ///     p.alloc(1),
    ///     p.alloc(2),
    /// ];
    ///
    /// for r in unsafe { p.borrow_batch_mut_unchecked(ids.iter().copied()) } {
    ///     // We can be certain that none of the items are [`None`]
    ///     // because we just allocated them.
    ///     *r.unwrap() *= 2;
    /// }
    ///
    /// for (i, id) in ids.into_iter().enumerate() {
    ///     assert_eq!(p.get(id), Some(&(i * 2)));
    /// }
    /// ```
    #[must_use]
    pub unsafe fn borrow_batch_mut_unchecked<'a, S>(
        &'a mut self,
        ids: S,
    ) -> impl Iterator<Item = Option<&'a mut T>>
    where
        S: IntoIterator<Item = Id>,
    {
        let ptr = self as *mut Self;

        ids.into_iter().map(move |id| unsafe { (*ptr).get_mut(id) })
    }

    /// Explicitly drop an id's value and mark it as uninitialized.
    /// This operation is safe, and will silently return without doing
    /// anything if `id` is invalid or uninit.
    pub fn free(&mut self, id: Id) {
        let Some(bucket) = self.buckets.get_mut(id.0 as usize) else {
            return;
        };

        if bucket.drop_index(id.1 as usize) {
            self.free_slots.push(id);
        }
    }

    /// Move an item out from the pool and own it.
    /// If the id is not valid or is associated with an uninitialized value,
    /// the function will return [`None`].
    #[must_use = "if you dont care about the value consider using `Pool::free`"]
    pub fn take(&mut self, id: Id) -> Option<T> {
        let v = self.buckets.get_mut(id.0 as usize)?.take(id.1 as usize);

        if v.is_some() {
            self.free_slots.push(id);
        }

        v
    }
}

impl<T> Index<Id> for Pool<T> {
    type Output = T;

    fn index(&self, index: Id) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> IndexMut<Id> for Pool<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn drop_on_free() {
        let count = Arc::new(AtomicUsize::new(0));

        let mut pool = Pool::new();
        let id = pool.alloc(DropCounter(count.clone()));

        assert_eq!(count.load(Ordering::Relaxed), 0);

        pool.free(id);

        assert_eq!(count.load(Ordering::Relaxed), 1);

        drop(pool);

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn drop_on_pool_drop() {
        let count = Arc::new(AtomicUsize::new(0));

        let mut pool = Pool::new();

        let _ = pool.alloc(DropCounter(count.clone()));
        let _ = pool.alloc(DropCounter(count.clone()));

        drop(pool);

        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn drop_contiguous_alloc() {
        let count = Arc::new(AtomicUsize::new(0));

        let mut pool = Pool::new();

        let (ids, _) =
            pool.alloc_contiguous(std::iter::repeat_with(|| DropCounter(count.clone())).take(23));

        assert_eq!(ids.len(), 23);

        for id in ids {
            pool.free(id);
        }

        assert_eq!(count.load(Ordering::Relaxed), 23);

        drop(pool);

        assert_eq!(count.load(Ordering::Relaxed), 23);
    }

    #[test]
    #[should_panic = "overlapping mutable borrow"]
    fn batch_borrow_violaton() {
        let mut pool = Pool::new();

        let id = pool.alloc(23);

        let _ = pool.borrow_batch_mut([id, id]);
    }

    #[test]
    fn take_doesnt_drop() {
        let count = Arc::new(AtomicUsize::new(0));

        let mut pool = Pool::new();

        let id = pool.alloc(DropCounter(count.clone()));
        let counter = pool.take(id).unwrap();

        assert!(pool.take(id).is_none());

        assert_eq!(count.load(Ordering::Relaxed), 0);

        drop(pool);

        assert_eq!(count.load(Ordering::Relaxed), 0);

        drop(counter);

        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
