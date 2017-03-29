#![feature(alloc, heap_api, process_abort, core_intrinsics)]

extern crate alloc;

use std::{fmt, ptr, mem, slice};
use std::ops::{Deref, DerefMut};
use alloc::heap;
use std::marker::PhantomData;
use std::cmp::*;
use std::hash::*;
use std::borrow::*;



/// The header of a ThinVec
struct Header {
    len: usize,
    cap: usize,
}

impl Header {
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
static EMPTY_HEADER: Header = Header { len: 0, cap: 0 };


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
        alloc_align - header_size
    } else {
        0
    }
}

fn alloc_align<T>() -> usize {
    max(mem::align_of::<T>(), mem::align_of::<Header>())
}

fn header_with_capacity<T>(cap: usize) -> *mut Header {
    unsafe {
        let header = heap::allocate(
            alloc_size::<T>(cap), 
            alloc_align::<T>(),
        ) as *mut Header; 

        if header.is_null() { oom() }

        // "Infinite" capacity for zero-sized types:
        (*header).cap = if mem::size_of::<T>() == 0 { !0 } else { cap };
        (*header).len = 0;

        header
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
/// * `size_of::<ThinVec<T>>()` == `size_of::<Option<ThinVec<T>>>()` (TODO) 
///
/// Properties of Vec that aren't preserved:
/// * `ThinVec<T>` can't ever be zero-cost roundtripped to a `Box<[T]>`, `String`, or `*mut T`
/// * `from_raw_parts` doesn't exist
/// * ThinVec currently doesn't bother to not-allocate for Zero Sized Types (e.g. `ThinVec<()>`),
///   but it could be done if someone cared enough to implement it.
pub struct ThinVec<T> {
    ptr: *const Header,
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
/// }
/// ```
#[macro_export]
macro_rules! thin_vec {
    /* TODO
    ($elem:expr; $n:expr) => (
        $crate::ThinVec::from_elem($elem, $n)
    );
    */
    ($($x:expr),*) => ({
        // TODO: Change this to work without cloning the elements.
        let mut vec = $crate::ThinVec::new();
        vec.extend_from_slice(&[$($x),*]);
        vec
    });
    ($($x:expr,)*) => (thin_vec![$($x),*])
}

impl<T> ThinVec<T> {
    pub fn new() -> ThinVec<T> {
        ThinVec {
            ptr: &EMPTY_HEADER,
            boo: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize) -> ThinVec<T> {
        ThinVec { 
            ptr: header_with_capacity::<T>(cap), 
            boo: PhantomData 
        }
    }

    // Accessor conveniences

    fn ptr(&self) -> *mut Header { self.ptr as *mut _ }
    fn header(&self) -> &Header { unsafe { &*self.ptr } }
    fn header_mut(&mut self) -> &mut Header { unsafe { &mut *self.ptr() } }
    fn data_raw(&self) -> *mut T { self.header().data() }

    pub fn len(&self) -> usize { self.header().len }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn capacity(&self) -> usize { self.header().cap }
    pub unsafe fn set_len(&mut self, len: usize) { self.header_mut().len = len }

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
            if std::intrinsics::needs_drop::<T>() {
                for x in &mut self[len..] {
                    ptr::drop_in_place(x)
                }
            }
            self.set_len(len);
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            if std::intrinsics::needs_drop::<T>() {
                for x in &mut self[..] {
                    ptr::drop_in_place(x);
                }
            }

            self.set_len(0)
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
            unsafe {
                self.reallocate(new_cap);
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
    
    pub fn append(&mut self, other: &mut ThinVec<T>) {
        // TODO
        // self.extend(other.drain())
    }

    /* TODO: RangeArgument is a pain
    pub fn drain<R>(&mut self, range: R) -> Drain<T> where R: RangeArgument<usize> {
        // TODO
    }
    */

    unsafe fn deallocate(&mut self) {
        if self.capacity() > 0 {
            heap::deallocate(self.ptr as *mut u8, 
                alloc_size::<T>(self.capacity()),
                alloc_align::<T>());
        }
    }

    /// Resize the buffer and update its capacity, without changing the length.
    /// Unsafe because it can cause length to be greater than capacity.
    unsafe fn reallocate(&mut self, new_cap: usize) {
        let old_cap = self.capacity();
        if old_cap == 0 {
            self.ptr = header_with_capacity::<T>(new_cap);
        } else {
            self.ptr = heap::reallocate(self.ptr() as *mut u8,
                                        alloc_size::<T>(old_cap),
                                        alloc_size::<T>(new_cap),
                                        alloc_align::<T>()) as *mut Header;
            if self.ptr.is_null() {
                oom()
            }
            self.header_mut().cap = new_cap;
        }
    }
}

impl<T: Clone> ThinVec<T> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        // TODO
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
            if std::intrinsics::needs_drop::<T>() {
                ptr::drop_in_place(&mut self [..]);
            }
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

pub struct Drain<'a, T: 'a> {
    vec: &'a mut ThinVec<T>,
    start: usize,
    end: usize,
    // TODO
}

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
            if std::intrinsics::needs_drop::<T>() {
                let mut vec = mem::replace(&mut self.vec, ThinVec::new());
                ptr::drop_in_place(&mut vec[self.start..]);
                vec.set_len(0)
            }
        }
    } 
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // TODO
    }
}

// TODO: a million Index impls
// TODO?: a million Cmp<[T; n]> impls

#[cfg(test)]
mod tests {
    use super::ThinVec;

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
}
