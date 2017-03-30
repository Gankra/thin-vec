#![feature(alloc, heap_api, process_abort, core_intrinsics)]

extern crate alloc;

use std::{ptr, mem, slice};
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
            // Don't do `GEP [inbounds]` for high alignment so EMPTY_HEADER is safe
            ptr.wrapping_offset((header_size + padding) as isize) as *mut T
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
    let header_size =  mem::size_of::<Header>();

    if header_size % alloc_align == 0 {
        0
    } else {
        alloc_align - header_size % alloc_align
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
        
        (*header).cap = cap;
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
        self.reserve_one_more();

        let old_len = self.len();
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
        self.reserve_one_more();
        
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

    pub fn reserve(&mut self, additional: usize) {
        // TODO
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        // TODO
    }

    pub fn shrink_to_fit(&mut self) {
        // TODO
    }

    pub fn retain<F>(&mut self, f: F) where F: FnMut(&T) -> bool { 
        // TODO
    }
    
    pub fn dedup_by_key<F, K>(&mut self, key: F) where F: FnMut(&mut T) -> K, K: PartialEq<K> {
        // TODO
    }
    
    pub fn dedup_by<F>(&mut self, same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        // TODO
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

    fn reserve_one_more(&mut self) {
        // TODO: capacity overflow logic?
        let old_cap = self.capacity();
        let old_len = self.len();

        if old_cap <= old_len {
            let elem_size = mem::size_of::<T>();

            // "Infinite" capacity for ZSTs
            let new_cap = if elem_size == 0 { !0 } else 
                          if old_cap == 0 { 4 } else 
                          { 2 * old_cap };

            unsafe {
                let old_data = self.data_raw();

                let new_header = header_with_capacity::<T>(new_cap);
                ptr::copy_nonoverlapping(old_data, (&*new_header).data::<T>(), old_len);
                self.deallocate();
                self.ptr = new_header;
                self.set_len(old_len);
            }
        }
    }

    unsafe fn deallocate(&mut self) {
        if self.capacity() > 0 {
            heap::deallocate(self.ptr as *mut u8, 
                alloc_size::<T>(self.capacity()),
                alloc_align::<T>());
        }
    } 
}

impl<T: Clone> ThinVec<T> {
    fn resize(&mut self, new_len: usize, value: T) {
        // TODO
    }
    
    fn extend_from_slice(&mut self, other: &[T]) {
        self.extend(other.iter().cloned())
    }
}

impl<T: PartialEq> ThinVec<T> {
    fn dedup(&mut self) {
        // TODO
    }
}

impl<T> Drop for ThinVec<T> {
    fn drop(&mut self) {
        unsafe {
            if std::intrinsics::needs_drop::<T>() {
                for x in &mut self[..] {
                    ptr::drop_in_place(x);
                }
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



impl<T> Hash for ThinVec<T> where T: Hash {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        for x in self {
            x.hash(state)
        }
    }
}

impl<T> PartialOrd<ThinVec<T>> for ThinVec<T> where T: PartialOrd<T> {
    fn partial_cmp(&self, other: &ThinVec<T>) -> Option<Ordering> {
        for (x, y) in self.iter().zip(other.iter()) {
            match x.partial_cmp(y) { 
                Some(Ordering::Equal) => { continue; }
                ordering => { return ordering; }
            }
        }

        // Whoever stopped first is the smaller one
        Some(self.len().cmp(&other.len()))
    }
}

impl<T> Ord for ThinVec<T> where T: Ord {
    fn cmp(&self, other: &ThinVec<T>) -> Ordering {
        for (x, y) in self.iter().zip(other.iter()) {
            match x.cmp(y) { 
                Ordering::Equal => { continue; }
                ordering => { return ordering; }
            }
        }

        // Whoever stopped first is the smaller one
        self.len().cmp(&other.len())
    }
}

impl<T> PartialEq for ThinVec<T> where T: PartialEq {
    fn eq(&self, other: &ThinVec<T>) -> bool {
        if self.len() == other.len() {
            for (x, y) in self.iter().zip(other.iter()) {
                if x != y { return false }
            }
            true
        } else {
            false
        }
    }
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
                for x in &mut vec[self.start..] {
                    ptr::drop_in_place(x)
                }
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
        let v = ThinVec::<u8>::new();
    }
}

// TODO: steal Vec's tests
fn main() {
    let mut vec = ThinVec::new();
    vec.push(0);
    vec.push(1);

    println!("{:?}", vec.pop());
    println!("{:?}", vec.pop());
    println!("{:?}", vec.pop());
}

