ThinVec is a Vec that stores its length and capacity inline, making it take up
less space. Currently this crate mostly exists to facilitate gecko ffi. The
crate isn't quite ready for use elsewhere, as it currently unconditionally
uses the libc allocator.

You may also want to check out [HeaderVec](https://github.com/rust-cv/header-vec),
which, in addition, allows you to put a header type of your choice at the
beginning of the allocation along with the width and capacity.