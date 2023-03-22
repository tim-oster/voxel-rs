use std::ops::{Deref, DerefMut};

macro_rules! impl_constructor {
    (
        $base: ident,
        $target: ident $( :: $path : ident )*,
        $($args: ident),+
    ) => {
        impl<T> $base<T> {
            #[allow(dead_code)]
            pub fn new($($args: T),+) -> $base<T> {
                $base($target$(::$path)*::<T>::new($($args),+))
            }
        }
    };
}

macro_rules! impl_deref {
    (
        $base: ident,
        $target: ident $( :: $path : ident )*
    ) => {
        impl<T> Deref for $base<T> {
            type Target = $target$(::$path)*<T>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<T> DerefMut for $base<T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}

#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedVec3<T>(pub cgmath::Vector3<T>);

impl_constructor!(AlignedVec3, cgmath::Vector3, x, y, z);
impl_deref!(AlignedVec3, cgmath::Vector3);

#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedVec4<T>(pub cgmath::Vector4<T>);

impl_constructor!(AlignedVec4, cgmath::Vector4, x, y, z, w);
impl_deref!(AlignedVec4, cgmath::Vector4);

#[repr(align(8))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedPoint2<T>(pub cgmath::Point2<T>);

impl_constructor!(AlignedPoint2, cgmath::Point2, x, y);
impl_deref!(AlignedPoint2, cgmath::Point2);

#[repr(align(16))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedPoint3<T>(pub cgmath::Point3<T>);

impl_constructor!(AlignedPoint3, cgmath::Point3, x, y, z);
impl_deref!(AlignedPoint3, cgmath::Point3);

#[repr(align(4))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AlignedBool(pub u32);

impl Into<bool> for AlignedBool {
    fn into(self) -> bool {
        self.0 != 0
    }
}

impl From<bool> for AlignedBool {
    fn from(value: bool) -> Self {
        AlignedBool(value as u32)
    }
}

#[macro_export]
macro_rules! assert_float_eq {
        ($expected: expr, $actual: expr) => {
            assert_float_eq!($expected, $actual, 1e-5)
        };
        ($expected: expr, $actual: expr, $eps: literal) => {
            {
                let diff = ($expected - $actual).abs();
                assert!(diff < $eps, "|{} - {}| = {} >= {}", $expected, $actual, diff, $eps);
                $expected
            }
        };
    }

#[macro_export]
macro_rules! gl_check_error {
    () => {
        crate::graphics::macros::gl_check_error_(file!(), line!())
    };
}
#[macro_export]
macro_rules! gl_assert_no_error {
    () => {
        assert!(!crate::graphics::macros::gl_check_error_(file!(), line!()))
    };
}
pub fn gl_check_error_(file: &str, line: u32) -> bool {
    let mut error_code = unsafe { gl::GetError() };
    if error_code == gl::NO_ERROR {
        return false;
    }

    while error_code != gl::NO_ERROR {
        let error = match error_code {
            gl::INVALID_ENUM => "INVALID_ENUM",
            gl::INVALID_VALUE => "INVALID_VALUE",
            gl::INVALID_OPERATION => "INVALID_OPERATION",
            gl::STACK_OVERFLOW => "STACK_OVERFLOW",
            gl::STACK_UNDERFLOW => "STACK_UNDERFLOW",
            gl::OUT_OF_MEMORY => "OUT_OF_MEMORY",
            gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
            _ => "unknown GL error code",
        };

        println!("{} | {} ({})", error, file, line);

        error_code = unsafe { gl::GetError() };
    }

    true
}
