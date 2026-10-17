use std::ops::{Add, Div, Mul, Sub};

use crate::kernel::{Cols, Rows};
use crate::simd::{self, Tile};

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Element type of the input sequences: `f32` or `f64`.
///
/// Frame distances are computed in this type; the DP accumulates in `f64`.
pub trait Float:
    private::Sealed
    + Copy
    + Send
    + Sync
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const INFINITY: Self;

    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    /// `self * b + c`, fused where the target has native FMA.
    fn fma(self, b: Self, c: Self) -> Self;

    /// Column padding of the transposed column-side sequence, and the
    /// micro-kernel's column block width (one SIMD-friendly register tile).
    const JB: usize;

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn fill_tile(a: &Rows<Self>, b: &Cols<Self>, i0: usize, l: usize, h: usize, tile: &mut [Self], tw: usize, cosine: bool);
}

macro_rules! impl_float {
    ($t:ty, $jb:expr, $fill:path) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const INFINITY: Self = <$t>::INFINITY;
            const JB: usize = $jb;

            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }

            #[inline(always)]
            fn fma(self, b: Self, c: Self) -> Self {
                // Without native FMA, mul_add is a slow libm call: use separate ops.
                #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
                {
                    self.mul_add(b, c)
                }
                #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
                {
                    self * b + c
                }
            }

            fn fill_tile(a: &Rows<Self>, b: &Cols<Self>, i0: usize, l: usize, h: usize, tile: &mut [Self], tw: usize, cosine: bool) {
                $fill(Tile { a, b, i0, l, h, tile, tw }, cosine)
            }
        }
    };
}

// 512 bits per accumulator row: four 128-bit or two 256-bit registers.
impl_float!(f64, 8, simd::fill_tile_f64);
impl_float!(f32, 16, simd::fill_tile_f32);
