#![cfg_attr(feature = "unstable", feature(shared))]

mod range;

use std::{fmt, ptr, mem, slice};
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::cmp::*;
use std::hash::*;
use std::borrow::*;
use range::RangeArgument;

// Heap shimming because reasons. This doesn't unfortunately match the heap api
// right now because reasons.
mod heap;

// Shared shimming because reasons.
#[cfg(feature = "unstable")]
use std::ptr::Shared;
#[cfg(not(feature = "unstable"))]
mod shared;
#[cfg(not(feature = "unstable"))]
use shared::Shared;

#[cfg(not(feature = "gecko-ffi"))]
type SizeType = usize;
#[cfg(feature = "gecko-ffi")]
type SizeType = u32;

#[cfg(feature = "gecko-ffi")]
const AUTO_MASK: u32 = 1 << 31;
#[cfg(feature = "gecko-ffi")]
const CAP_MASK: u32 = !AUTO_MASK;

#[cfg(not(feature = "gecko-ffi"))]
#[inline(always)]
fn assert_size(x: usize) -> SizeType { x }

#[cfg(feature = "gecko-ffi")]
#[inline]
fn assert_size(x: usize) -> SizeType {
    if x > std::i32::MAX as usize {
        panic!("nsTArray size may not exceed the capacity of a 32-bit sized int");
    }
    x as SizeType
}

/// The header of a ThinVec
#[repr(C)]
struct Header {
    _len: SizeType,
    _cap: SizeType,
}

impl Header {
    fn len(&self) -> usize {
        self._len as usize
    }

    #[cfg(feature = "gecko-ffi")]
    fn cap(&self) -> usize {
        (self._cap & CAP_MASK) as usize
    }

    #[cfg(not(feature = "gecko-ffi"))]
    fn cap(&self) -> usize {
        self._cap as usize
    }

    fn set_len(&mut self, len: usize) {
        self._len = assert_size(len);
    }

    #[cfg(feature = "gecko-ffi")]
    fn set_cap(&mut self, cap: usize) {
        debug_assert!(cap & (CAP_MASK as usize) == cap);
        debug_assert!(!self.uses_stack_allocated_buffer());
        self._cap = assert_size(cap) & CAP_MASK;
    }

    #[cfg(feature = "gecko-ffi")]
    fn uses_stack_allocated_buffer(&self) -> bool {
        self._cap & AUTO_MASK != 0
    }

    #[cfg(not(feature = "gecko-ffi"))]
    fn set_cap(&mut self, cap: usize) {
        self._cap = assert_size(cap);
    }

    fn data<T>(&self) -> *mut T {
        let header_size = mem::size_of::<Header>();
        let padding = padding::<T>();

        let ptr = self as *const Header as *mut Header as *mut u8;

        unsafe {
            if padding > 0 {
                // Don't do `GEP [inbounds]` for high alignment so EMPTY_HEADER is safe
                ptr.wrapping_offset((header_size + padding) as isize) as *mut T
            } else {
                ptr.offset(header_size as isize) as *mut T
            }
        }
    }
}

/// Singleton that all empty collections share.
/// Note: can't store non-zero ZSTs, we allocate in that case. We could
/// optimize everything to not do that (basically, make ptr == len and branch
/// on size == 0 in every method), but it's a bunch of work for something that
/// doesn't matter much.
#[cfg(not(feature = "gecko-internal"))]
static EMPTY_HEADER: Header = Header { _len: 0, _cap: 0 };

#[cfg(feature = "gecko-internal")]
extern {
    #[link_name = "sEmptyTArrayHeader"]
    static EMPTY_HEADER: Header;
}


// TODO: overflow checks everywhere

// Utils

fn oom() -> ! { std::process::abort() }

fn alloc_size<T>(cap: usize) -> usize {
    // Compute "real" header size with pointer math
    let header_size = mem::size_of::<Header>();
    let elem_size = mem::size_of::<T>();
    let padding = padding::<T>();

    // TODO: care about isize::MAX overflow?
    let data_size = elem_size.checked_mul(cap).expect("capacity overflow");

    data_size.checked_add(header_size + padding).expect("capacity overflow")
}

fn padding<T>() -> usize {
    let alloc_align = alloc_align::<T>();
    let header_size = mem::size_of::<Header>();

    if alloc_align > header_size {
        if cfg!(feature = "gecko-ffi") {
            panic!("nsTArray does not handle alignment above > {} correctly",
                   header_size);
        }
        alloc_align - header_size
    } else {
        0
    }
}

fn alloc_align<T>() -> usize {
    max(mem::align_of::<T>(), mem::align_of::<Header>())
}

fn header_with_capacity<T>(cap: usize) -> Shared<Header> {
    debug_assert!(cap > 0);
    unsafe {
        let header = heap::allocate(
            alloc_size::<T>(cap),
            alloc_align::<T>(),
        ) as *mut Header;

        if header.is_null() { oom() }

        // "Infinite" capacity for zero-sized types:
        (*header).set_cap(if mem::size_of::<T>() == 0 { !0 } else { cap });
        (*header).set_len(0);

        Shared::new_unchecked(header)
    }
}



/// ThinVec is exactly the same as Vec, except that it stores its `len` and `capacity` in the buffer
/// it allocates.
///
/// This makes the memory footprint of ThinVecs lower; notably in cases where space is reserved for
/// a non-existence ThinVec<T>. So `Vec<ThinVec<T>>` and `Option<ThinVec<T>>::None` will waste less
/// space. Being pointer-sized also means it can be passed/stored in registers.
///
/// Of course, any actually constructed ThinVec will theoretically have a bigger allocation, but
/// the fuzzy nature of allocators means that might not actually be the case.
///
/// Properties of Vec that are preserved:
/// * `ThinVec::new()` doesn't allocate (it points to a statically allocated singleton)
/// * reallocation can be done in place
/// * `size_of::<ThinVec<T>>()` == `size_of::<Option<ThinVec<T>>>()`
///   * NOTE: This is only possible when the `unstable` feature is used.
///
/// Properties of Vec that aren't preserved:
/// * `ThinVec<T>` can't ever be zero-cost roundtripped to a `Box<[T]>`, `String`, or `*mut T`
/// * `from_raw_parts` doesn't exist
/// * ThinVec currently doesn't bother to not-allocate for Zero Sized Types (e.g. `ThinVec<()>`),
///   but it could be done if someone cared enough to implement it.
#[cfg_attr(feature = "gecko-ffi", repr(C))]
pub struct ThinVec<T> {
    ptr: Shared<Header>,
    boo: PhantomData<T>,
}


/// Creates a `ThinVec` containing the arguments.
///
/// ```
/// #[macro_use] extern crate thin_vec;
///
/// fn main() {
///     let v = thin_vec![1, 2, 3];
///     assert_eq!(v.len(), 3);
///     assert_eq!(v[0], 1);
///     assert_eq!(v[1], 2);
///     assert_eq!(v[2], 3);
///
///     let v = vec![1; 3];
///     assert_eq!(v, [1, 1, 1]);
/// }
/// ```
#[macro_export]
macro_rules! thin_vec {
    (@UNIT $($t:tt)*) => (());

    ($elem:expr; $n:expr) => ({
        let mut vec = $crate::ThinVec::new();
        vec.resize($n, $elem);
        vec
    });
    ($($x:expr),*) => ({
        let len = [$(thin_vec!(@UNIT $x)),*].len();
        let mut vec = $crate::ThinVec::with_capacity(len);
        $(vec.push($x);)*
        vec
    });
    ($($x:expr,)*) => (thin_vec![$($x),*]);
}

impl<T> ThinVec<T> {
    pub fn new() -> ThinVec<T> {
        unsafe {
            ThinVec {
                ptr: Shared::new_unchecked(&EMPTY_HEADER
                                           as *const Header
                                           as *mut Header),
                boo: PhantomData,
            }
        }
    }

    pub fn with_capacity(cap: usize) -> ThinVec<T> {
        if cap == 0 {
            ThinVec::new()
        } else {
            ThinVec {
                ptr: header_with_capacity::<T>(cap),
                boo: PhantomData,
            }
        }
    }

    // Accessor conveniences

    fn ptr(&self) -> *mut Header { self.ptr.as_ptr() }
    fn header(&self) -> &Header { unsafe { self.ptr.as_ref() } }
    fn data_raw(&self) -> *mut T { self.header().data() }

    // This is unsafe when the header is EMPTY_HEADER.
    unsafe fn header_mut(&mut self) -> &mut Header { &mut *self.ptr() }

    pub fn len(&self) -> usize { self.header().len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn capacity(&self) -> usize { self.header().cap() }
    pub unsafe fn set_len(&mut self, len: usize) { self.header_mut().set_len(len) }

    pub fn push(&mut self, val: T) {
        let old_len = self.len();
        if old_len == self.capacity() {
            self.reserve(1);
        }
        unsafe {
            ptr::write(self.data_raw().offset(old_len as isize), val);
            self.set_len(old_len + 1);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let old_len = self.len();
        if old_len == 0 { return None }

        unsafe {
            self.set_len(old_len - 1);
            Some(ptr::read(self.data_raw().offset(old_len as isize - 1)))
        }
    }

    pub fn insert(&mut self, idx: usize, elem: T) {
        let old_len = self.len();

        assert!(idx <= old_len, "Index out of bounds");
        if old_len == self.capacity() {
            self.reserve(1);
        }
        unsafe {
            let ptr = self.data_raw();
            ptr::copy(ptr.offset(idx as isize), ptr.offset(idx as isize + 1), old_len - idx);
            ptr::write(ptr.offset(idx as isize), elem);
            self.set_len(old_len + 1);
        }
    }

    pub fn remove(&mut self, idx: usize) -> T {
        let old_len = self.len();

        assert!(idx < old_len, "Index out of bounds");

        unsafe {
            self.set_len(old_len - 1);
            let ptr = self.data_raw();
            let val = ptr::read(self.data_raw().offset(idx as isize));
            ptr::copy(ptr.offset(idx as isize + 1), ptr.offset(idx as isize),
                      old_len - idx - 1);
            val
        }
    }

    pub fn swap_remove(&mut self, idx: usize) -> T {
        let old_len = self.len();

        assert!(idx < old_len, "Index out of bounds");

        unsafe {
            let ptr = self.data_raw();
            ptr::swap(ptr.offset(idx as isize), ptr.offset(old_len as isize - 1));
            self.set_len(old_len - 1);
            ptr::read(ptr.offset(old_len as isize - 1))
        }
    }

    pub fn truncate(&mut self, len: usize) {
        let old_len = self.len();

        assert!(len <= old_len, "Can't truncate to a larger len than the current one");

        unsafe {
            ptr::drop_in_place(&mut self [len..]);
            self.set_len(len);
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            ptr::drop_in_place(&mut self[..]);
            self.set_len(0);
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.data_raw(), self.len())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.data_raw(), self.len())
        }
    }

    /// Reserve capacity for at least `additional` more elements to be inserted.
    ///
    /// May reserve more space than requested, to avoid frequent reallocations.
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// Re-allocates only if `self.capacity() < self.len() + additional`.
    #[cfg(not(feature = "gecko-ffi"))]
    pub fn reserve(&mut self, additional: usize) {
        let len = self.len();
        let old_cap = self.capacity();
        let min_cap = len.checked_add(additional).expect("capacity overflow");
        if min_cap <= old_cap {
            return
        }
        // Ensure the new capacity is at least double, to guarantee exponential growth.
        let double_cap = if old_cap == 0 {
            // skip to 4 because tiny Vecs are dumb; but not if that would cause overflow
            if mem::size_of::<T>() > (!0) / 8 { 1 } else { 4 }
        } else {
            old_cap.saturating_mul(2)
        };
        let new_cap = max(min_cap, double_cap);
        unsafe {
            self.reallocate(new_cap);
        }
    }

    /// Reserve capacity for at least `additional` more elements to be inserted.
    ///
    /// This method mimics the growth algorithm used by the C++ implementation
    /// of nsTArray.
    #[cfg(feature = "gecko-ffi")]
    pub fn reserve(&mut self, additional: usize) {
        let elem_size = mem::size_of::<T>();

        let len = self.len();
        let old_cap = self.capacity();
        let min_cap = len.checked_add(additional).expect("capacity overflow");
        if min_cap <= old_cap {
            return
        }

        let min_cap_bytes = assert_size(min_cap)
            .checked_mul(assert_size(elem_size))
            .and_then(|x| x.checked_add(assert_size(mem::size_of::<Header>())))
            .unwrap();

        // Perform some checked arithmetic to ensure all of the numbers we
        // compute will end up in range.
        let will_fit = min_cap_bytes.checked_mul(2).is_some();
        if !will_fit {
            panic!("Exceeded maximum nsTArray size");
        }

        const SLOW_GROWTH_THRESHOLD: usize = 8 * 1024 * 1024;

        let bytes = if min_cap > SLOW_GROWTH_THRESHOLD {
            // Grow by a minimum of 1.125x
            let old_cap_bytes = old_cap * elem_size + mem::size_of::<Header>();
            let min_growth = old_cap_bytes + (old_cap_bytes >> 3);
            let growth = max(min_growth, min_cap_bytes as usize);

            // Round up to the next megabyte.
            const MB: usize = 1 << 20;
            MB * ((growth + MB - 1) / MB)
        } else {
            // Try to allocate backing buffers in powers of two.
            min_cap_bytes.next_power_of_two() as usize
        };

        let cap = (bytes - std::mem::size_of::<Header>()) / elem_size;
        unsafe {
            self.reallocate(cap);
        }
    }

    /// Reserves the minimum capacity for `additional` more elements to be inserted.
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// Re-allocates only if `self.capacity() < self.len() + additional`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let new_cap = self.len().checked_add(additional).expect("capacity overflow");
        let old_cap = self.capacity();
        if new_cap > old_cap {
            unsafe {
                self.reallocate(new_cap);
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        let old_cap = self.capacity();
        let new_cap = self.len();
        if new_cap < old_cap {
            if new_cap == 0 {
                *self = ThinVec::new();
            } else {
                unsafe {
                    self.reallocate(new_cap);
                }
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thin_vec;
    /// # fn main() {
    /// let mut vec = thin_vec![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// # }
    /// ```
    pub fn retain<F>(&mut self, mut f: F) where F: FnMut(&T) -> bool {
        let len = self.len();
        let mut del = 0;
        {
            let v = &mut self[..];

            for i in 0..len {
                if !f(&v[i]) {
                    del += 1;
                } else if del > 0 {
                    v.swap(i - del, i);
                }
            }
        }
        if del > 0 {
            self.truncate(len - del);
        }
    }

    /// Removes consecutive elements in the vector that resolve to the same key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thin_vec;
    /// # fn main() {
    /// let mut vec = thin_vec![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// # }
    /// ```
    pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq<K> {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    /// Removes consecutive elements in the vector according to a predicate.
    ///
    /// The `same_bucket` function is passed references to two elements from the vector, and
    /// returns `true` if the elements compare equal, or `false` if they do not. Only the first
    /// of adjacent equal items is kept.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thin_vec;
    /// # fn main() {
    /// use std::ascii::AsciiExt;
    ///
    /// let mut vec = thin_vec!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// # }
    /// ```
    pub fn dedup_by<F>(&mut self, mut same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        // See the comments in `Vec::dedup` for a detailed explanation of this code.
        unsafe {
            let ln = self.len();
            if ln <= 1 {
                return;
            }

            // Avoid bounds checks by using raw pointers.
            let p = self.as_mut_ptr();
            let mut r: usize = 1;
            let mut w: usize = 1;

            while r < ln {
                let p_r = p.offset(r as isize);
                let p_wm1 = p.offset((w - 1) as isize);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
                r += 1;
            }

            self.truncate(w);
        }
    }

    pub fn split_off(&mut self, at: usize) -> ThinVec<T> {
        let old_len = self.len();
        let new_vec_len = old_len - at;

        assert!(at <= old_len, "Index out of bounds");

        unsafe {
            let mut new_vec = ThinVec::with_capacity(new_vec_len);

            ptr::copy_nonoverlapping(self.data_raw().offset(at as isize),
                                     new_vec.data_raw(),
                                     new_vec_len);

            new_vec.set_len(new_vec_len);
            self.set_len(at);

            new_vec
        }
    }

    pub fn append(&mut self, _other: &mut ThinVec<T>) {
        // TODO
        // self.extend(other.drain())
    }

    /* TODO: RangeArgument is a pain
    pub fn drain<R>(&mut self, range: R) -> Drain<T> where R: RangeArgument<usize> {
        // TODO
    }
    */

    unsafe fn deallocate(&mut self) {
        if self.has_allocation() {
            heap::deallocate(self.ptr() as *mut u8,
                alloc_size::<T>(self.capacity()),
                alloc_align::<T>());
        }
    }

    /// Resize the buffer and update its capacity, without changing the length.
    /// Unsafe because it can cause length to be greater than capacity.
    unsafe fn reallocate(&mut self, new_cap: usize) {
        debug_assert!(new_cap > 0);
        if self.has_allocation() {
            let old_cap = self.capacity();
            let ptr = heap::reallocate(self.ptr() as *mut u8,
                                       alloc_size::<T>(old_cap),
                                       alloc_size::<T>(new_cap),
                                       alloc_align::<T>()) as *mut Header;
            if ptr.is_null() { oom() }
            (*ptr).set_cap(new_cap);
            self.ptr = Shared::new_unchecked(ptr);
        } else {
            self.ptr = header_with_capacity::<T>(new_cap);
        }
    }

    #[cfg(feature = "gecko-ffi")]
    #[inline]
    fn has_allocation(&self) -> bool {
        unsafe {
            self.ptr.as_ptr() as *const Header != &EMPTY_HEADER &&
                !self.ptr.as_ref().uses_stack_allocated_buffer()
        }
    }

    #[cfg(not(feature = "gecko-ffi"))]
    #[inline]
    fn has_allocation(&self) -> bool {
        self.ptr.as_ptr() as *const Header != &EMPTY_HEADER
    }
}

impl<T: Clone> ThinVec<T> {
    /// Resizes the `Vec` in-place so that `len()` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len()`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len()`, the `Vec` is simply truncated.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thin_vec;
    /// # fn main() {
    /// let mut vec = thin_vec!["hello"];
    /// vec.resize(3, "world");
    /// assert_eq!(vec, ["hello", "world", "world"]);
    ///
    /// let mut vec = thin_vec![1, 2, 3, 4];
    /// vec.resize(2, 0);
    /// assert_eq!(vec, [1, 2]);
    /// # }
    /// ```
    pub fn resize(&mut self, new_len: usize, value: T) {
        let old_len = self.len();

        if new_len > old_len {
            let additional = new_len - old_len;
            self.reserve(additional);
            for _ in 1..additional {
                self.push(value.clone());
            }
            // We can write the last element directly without cloning needlessly
            if additional > 0 {
                self.push(value);
            }
        } else if new_len < old_len {
            self.truncate(new_len);
        }
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.extend(other.iter().cloned())
    }
}

impl<T: PartialEq> ThinVec<T> {
    /// Removes consecutive repeated elements in the vector.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thin_vec;
    /// # fn main() {
    /// let mut vec = thin_vec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// # }
    /// ```
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }
}

impl<T> Drop for ThinVec<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(&mut self [..]);
            self.deallocate();
        }
    }
}

impl<T> Deref for ThinVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for ThinVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Borrow<[T]> for ThinVec<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> BorrowMut<[T]> for ThinVec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for ThinVec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Extend<T> for ThinVec<T> {
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item=T> {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for x in iter {
            self.push(x);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for ThinVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> Hash for ThinVec<T> where T: Hash {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        self[..].hash(state);
    }
}

impl<T> PartialOrd for ThinVec<T> where T: PartialOrd {
    #[inline]
    fn partial_cmp(&self, other: &ThinVec<T>) -> Option<Ordering> {
        self[..].partial_cmp(&other[..])
    }
}

impl<T> Ord for ThinVec<T> where T: Ord {
    #[inline]
    fn cmp(&self, other: &ThinVec<T>) -> Ordering {
        self[..].cmp(&other[..])
    }
}

impl<T, U> PartialEq<U> for ThinVec<T> where U: for<'a> PartialEq<&'a [T]> {
    fn eq(&self, other: &U) -> bool { *other == &self[..] }
    fn ne(&self, other: &U) -> bool { *other != &self[..] }
}

impl<T> Eq for ThinVec<T> where T: Eq {}

impl<T> IntoIterator for ThinVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter { vec: self, start: 0 }
    }
}

impl<'a, T> IntoIterator for &'a ThinVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ThinVec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> Clone for ThinVec<T> where T: Clone {
    fn clone(&self) -> ThinVec<T> {
        let mut new_vec = ThinVec::with_capacity(self.len());
        new_vec.extend(self.iter().cloned());
        new_vec
    }
}

impl<T> Default for ThinVec<T> {
    fn default() -> ThinVec<T> {
        ThinVec::new()
    }
}

pub struct IntoIter<T> {
    vec: ThinVec<T>,
    start: usize,
}

/*
pub struct Drain<'a, T: 'a> {
    vec: &'a mut ThinVec<T>,
    start: usize,
    end: usize,
    // TODO
}
*/

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.vec.len() {
            None
        } else {
            unsafe {
                let old_start = self.start;
                self.start += 1;
                Some(ptr::read(self.vec.data_raw().offset(old_start as isize)))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.vec.len() - self.start;
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.vec.len() {
            None
        } else {
            // FIXME?: extra bounds check
            self.vec.pop()
        }
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        unsafe {
            let mut vec = mem::replace(&mut self.vec, ThinVec::new());
            ptr::drop_in_place(&mut vec[self.start..]);
            vec.set_len(0)
        }
    }
}

/*
impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // TODO
    }
}
*/

// TODO: a million Index impls
// TODO?: a million Cmp<[T; n]> impls

#[cfg(test)]
mod tests {
    use super::ThinVec;

    #[test]
    fn test_size_of() {
        use std::mem::size_of;
        assert_eq!(size_of::<ThinVec<u8>>(), size_of::<&u8>());
        // We don't do this for reasons
        // assert_eq!(size_of::<Option<ThinVec<u8>>>(), size_of::<&u8>());
    }

    #[test]
    fn test_drop_empty() {
        ThinVec::<u8>::new();
    }

    #[test]
    fn test_partial_eq() {
        assert_eq!(thin_vec![0], thin_vec![0]);
        assert_ne!(thin_vec![0], thin_vec![1]);
        assert_eq!(thin_vec![1,2,3], vec![1,2,3]);
    }

    #[test]
    fn test_alloc() {
        let mut v = ThinVec::new();
        assert!(!v.has_allocation());
        v.push(1);
        assert!(v.has_allocation());
        v.pop();
        assert!(v.has_allocation());
        v.shrink_to_fit();
        assert!(!v.has_allocation());
        v.reserve(64);
        assert!(v.has_allocation());
        v = ThinVec::with_capacity(64);
        assert!(v.has_allocation());
        v = ThinVec::with_capacity(0);
        assert!(!v.has_allocation());
    }
}
