//! Python bindings. Argument normalisation (dtype/contiguity coercion,
//! metric names, n_jobs conventions) lives in the Python wrapper; these
//! functions expect C-contiguous 2-D arrays of a single dtype.

use cdtw_core::{Error, Float, Metric, Options, Seq};
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn options(metric: u8, window: Option<usize>) -> PyResult<Options> {
    let metric = match metric {
        0 => Metric::Euclidean,
        1 => Metric::Cosine,
        m => return Err(PyValueError::new_err(format!("unknown metric code {m}"))),
    };
    Ok(Options { metric, window })
}

fn seq<'a, T: Element>(arr: &'a PyReadonlyArray2<'_, T>) -> PyResult<Seq<'a, T>> {
    let shape = arr.shape();
    Seq::new(arr.as_slice()?, shape[0], shape[1]).map_err(value_error)
}

fn seqs<'a, T: Element>(arrs: &'a [PyReadonlyArray2<'_, T>]) -> PyResult<Vec<Seq<'a, T>>> {
    arrs.iter().map(seq).collect()
}

fn dtw_impl<T: Float + Element>(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, T>,
    b: PyReadonlyArray2<'_, T>,
    metric: u8,
    window: Option<usize>,
    n_jobs: Option<usize>,
) -> PyResult<f64> {
    let (a, b, opts) = (seq(&a)?, seq(&b)?, options(metric, window)?);
    py.detach(|| cdtw_core::dtw(a, b, &opts, n_jobs)).map_err(value_error)
}

fn dtw_path_impl<'py, T: Float + Element>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, T>,
    b: PyReadonlyArray2<'py, T>,
    metric: u8,
    window: Option<usize>,
    n_jobs: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray2<isize>>)> {
    let (a, b, opts) = (seq(&a)?, seq(&b)?, options(metric, window)?);
    let (cost, path) = py.detach(|| cdtw_core::dtw_path(a, b, &opts, n_jobs)).map_err(value_error)?;
    let flat: Vec<isize> = path.iter().flat_map(|&(i, j)| [i as isize, j as isize]).collect();
    Ok((cost, PyArray1::from_vec(py, flat).reshape([path.len(), 2])?))
}

fn cdist_impl<'py, T: Float + Element>(
    py: Python<'py>,
    a: Vec<PyReadonlyArray2<'py, T>>,
    b: Vec<PyReadonlyArray2<'py, T>>,
    metric: u8,
    window: Option<usize>,
    n_jobs: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (a, b, opts) = (seqs(&a)?, seqs(&b)?, options(metric, window)?);
    let out = py.detach(|| cdtw_core::cdist(&a, &b, &opts, n_jobs)).map_err(value_error)?;
    PyArray1::from_vec(py, out).reshape([a.len(), b.len()])
}

fn pdist_impl<'py, T: Float + Element>(
    py: Python<'py>,
    s: Vec<PyReadonlyArray2<'py, T>>,
    metric: u8,
    window: Option<usize>,
    n_jobs: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (s, opts) = (seqs(&s)?, options(metric, window)?);
    let out = py.detach(|| cdtw_core::pdist(&s, &opts, n_jobs)).map_err(value_error)?;
    Ok(PyArray1::from_vec(py, out))
}

macro_rules! typed {
    ($t:ty, $dtw:ident, $dtw_path:ident, $cdist:ident, $pdist:ident) => {
        #[pyfunction]
        fn $dtw(
            py: Python<'_>,
            a: PyReadonlyArray2<'_, $t>,
            b: PyReadonlyArray2<'_, $t>,
            metric: u8,
            window: Option<usize>,
            n_jobs: Option<usize>,
        ) -> PyResult<f64> {
            dtw_impl(py, a, b, metric, window, n_jobs)
        }

        #[pyfunction]
        fn $dtw_path<'py>(
            py: Python<'py>,
            a: PyReadonlyArray2<'py, $t>,
            b: PyReadonlyArray2<'py, $t>,
            metric: u8,
            window: Option<usize>,
            n_jobs: Option<usize>,
        ) -> PyResult<(f64, Bound<'py, PyArray2<isize>>)> {
            dtw_path_impl(py, a, b, metric, window, n_jobs)
        }

        #[pyfunction]
        fn $cdist<'py>(
            py: Python<'py>,
            a: Vec<PyReadonlyArray2<'py, $t>>,
            b: Vec<PyReadonlyArray2<'py, $t>>,
            metric: u8,
            window: Option<usize>,
            n_jobs: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray2<f64>>> {
            cdist_impl(py, a, b, metric, window, n_jobs)
        }

        #[pyfunction]
        fn $pdist<'py>(
            py: Python<'py>,
            s: Vec<PyReadonlyArray2<'py, $t>>,
            metric: u8,
            window: Option<usize>,
            n_jobs: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            pdist_impl(py, s, metric, window, n_jobs)
        }
    };
}

typed!(f64, dtw_f64, dtw_path_f64, cdist_f64, pdist_f64);
typed!(f32, dtw_f32, dtw_path_f32, cdist_f32, pdist_f32);

#[pymodule]
fn _cdtw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dtw_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_path_f64, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_path_f32, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_f64, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_f32, m)?)?;
    m.add_function(wrap_pyfunction!(pdist_f64, m)?)?;
    m.add_function(wrap_pyfunction!(pdist_f32, m)?)?;
    Ok(())
}
