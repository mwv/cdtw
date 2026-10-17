//! Minimal explicit SIMD layer for the distance micro-kernel.
//!
//! LLVM does not reliably SLP-vectorise the register-tiled kernel, so the
//! lanes are spelled out: NEON on aarch64, SSE2 (baseline) or AVX2+FMA
//! (runtime-detected) on x86-64, and a scalar fallback elsewhere.

use crate::float::Float;
use crate::kernel::{fill_tile, Cols, Rows};

/// A vector of `LANES` values of `T`. All methods are `unsafe` because they
/// may require target features; pointers must be valid for `LANES` values.
pub trait Simd<T>: Copy {
    const LANES: usize;
    unsafe fn zero() -> Self;
    unsafe fn splat(x: T) -> Self;
    unsafe fn load(p: *const T) -> Self;
    unsafe fn store(self, p: *mut T);
    unsafe fn sub(self, o: Self) -> Self;
    /// `self * b + c`
    unsafe fn mul_add(self, b: Self, c: Self) -> Self;
}

#[allow(dead_code)] // unused on aarch64/x86-64
#[derive(Clone, Copy)]
pub struct Scalar<T>(T);

impl<T: Float> Simd<T> for Scalar<T> {
    const LANES: usize = 1;
    #[inline(always)]
    unsafe fn zero() -> Self {
        Scalar(T::ZERO)
    }
    #[inline(always)]
    unsafe fn splat(x: T) -> Self {
        Scalar(x)
    }
    #[inline(always)]
    unsafe fn load(p: *const T) -> Self {
        Scalar(*p)
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut T) {
        *p = self.0
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        Scalar(self.0 - o.0)
    }
    #[inline(always)]
    unsafe fn mul_add(self, b: Self, c: Self) -> Self {
        Scalar(self.0.fma(b.0, c.0))
    }
}

macro_rules! simd_type {
    ($name:ident, $t:ty, $v:ty, $lanes:expr,
     zero: $zero:path, splat: $splat:path, load: $load:path, store: $store:path,
     sub: $sub:path, fma: |$a:ident, $b:ident, $c:ident| $fma:expr) => {
        #[derive(Clone, Copy)]
        pub struct $name($v);

        impl crate::simd::Simd<$t> for $name {
            const LANES: usize = $lanes;
            #[inline(always)]
            unsafe fn zero() -> Self {
                $name($zero())
            }
            #[inline(always)]
            unsafe fn splat(x: $t) -> Self {
                $name($splat(x))
            }
            #[inline(always)]
            unsafe fn load(p: *const $t) -> Self {
                $name($load(p))
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut $t) {
                $store(p, self.0)
            }
            #[inline(always)]
            unsafe fn sub(self, o: Self) -> Self {
                $name($sub(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn mul_add(self, b: Self, c: Self) -> Self {
                let ($a, $b, $c) = (self.0, b.0, c.0);
                $name($fma)
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    unsafe fn zero_f64() -> float64x2_t {
        vdupq_n_f64(0.0)
    }
    unsafe fn zero_f32() -> float32x4_t {
        vdupq_n_f32(0.0)
    }

    simd_type!(F64x2, f64, float64x2_t, 2,
        zero: zero_f64, splat: vdupq_n_f64, load: vld1q_f64, store: vst1q_f64,
        sub: vsubq_f64, fma: |a, b, c| vfmaq_f64(c, a, b));
    simd_type!(F32x4, f32, float32x4_t, 4,
        zero: zero_f32, splat: vdupq_n_f32, load: vld1q_f32, store: vst1q_f32,
        sub: vsubq_f32, fma: |a, b, c| vfmaq_f32(c, a, b));
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    // SSE2 is part of the x86-64 baseline; it has no FMA.
    simd_type!(F64x2, f64, __m128d, 2,
        zero: _mm_setzero_pd, splat: _mm_set1_pd, load: _mm_loadu_pd, store: _mm_storeu_pd,
        sub: _mm_sub_pd, fma: |a, b, c| _mm_add_pd(_mm_mul_pd(a, b), c));
    simd_type!(F32x4, f32, __m128, 4,
        zero: _mm_setzero_ps, splat: _mm_set1_ps, load: _mm_loadu_ps, store: _mm_storeu_ps,
        sub: _mm_sub_ps, fma: |a, b, c| _mm_add_ps(_mm_mul_ps(a, b), c));
    // Only used inside `#[target_feature(enable = "avx2,fma")]` functions.
    simd_type!(F64x4, f64, __m256d, 4,
        zero: _mm256_setzero_pd, splat: _mm256_set1_pd, load: _mm256_loadu_pd, store: _mm256_storeu_pd,
        sub: _mm256_sub_pd, fma: |a, b, c| _mm256_fmadd_pd(a, b, c));
    simd_type!(F32x8, f32, __m256, 8,
        zero: _mm256_setzero_ps, splat: _mm256_set1_ps, load: _mm256_loadu_ps, store: _mm256_storeu_ps,
        sub: _mm256_sub_ps, fma: |a, b, c| _mm256_fmadd_ps(a, b, c));

    /// AVX2+FMA available and not disabled with `CDTW_DISABLE_AVX2=1` (used
    /// to test the SSE2 path on AVX2 machines). Checked once per process.
    pub fn has_avx2_fma() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("CDTW_DISABLE_AVX2").is_none_or(|v| v.is_empty() || v == "0")
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
        })
    }
}

/// Arguments of one tile fill; see [`fill_tile`].
pub struct Tile<'s, 'a, T: Float> {
    pub a: &'s Rows<'a, T>,
    pub b: &'s Cols<T>,
    pub i0: usize,
    pub l: usize,
    pub h: usize,
    pub tile: &'s mut [T],
    pub tw: usize,
}

#[inline(always)]
fn select<T: Float, V: Simd<T>, const NV: usize>(t: Tile<'_, '_, T>, cosine: bool) {
    if cosine {
        fill_tile::<T, V, NV, true>(t.a, t.b, t.i0, t.l, t.h, t.tile, t.tw)
    } else {
        fill_tile::<T, V, NV, false>(t.a, t.b, t.i0, t.l, t.h, t.tile, t.tw)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_f64(t: Tile<'_, '_, f64>, cosine: bool) {
    select::<f64, x86::F64x4, 2>(t, cosine)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_f32(t: Tile<'_, '_, f32>, cosine: bool) {
    select::<f32, x86::F32x8, 2>(t, cosine)
}

// Tile widths: 8 f64 / 16 f32 columns = four 128-bit or two 256-bit vectors.

pub fn fill_tile_f64(t: Tile<'_, '_, f64>, cosine: bool) {
    #[cfg(target_arch = "aarch64")]
    select::<f64, neon::F64x2, 4>(t, cosine);
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        unsafe { avx2_f64(t, cosine) }
    } else {
        select::<f64, x86::F64x2, 4>(t, cosine)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    select::<f64, Scalar<f64>, 8>(t, cosine);
}

pub fn fill_tile_f32(t: Tile<'_, '_, f32>, cosine: bool) {
    #[cfg(target_arch = "aarch64")]
    select::<f32, neon::F32x4, 4>(t, cosine);
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        unsafe { avx2_f32(t, cosine) }
    } else {
        select::<f32, x86::F32x4, 4>(t, cosine)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    select::<f32, Scalar<f32>, 16>(t, cosine);
}
