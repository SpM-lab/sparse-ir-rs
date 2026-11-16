//! Public paths kept across the move of the SVE into `sparse-ir-sve`
//!
//! Each line fails to compile if its path is no longer re-exported.

#[test]
fn paths_from_before_the_split() {
    // `tsvd` is both a module and a function at the same path.
    let _ = sparse_ir::tsvd::<f64>;
    let _ = sparse_ir::tsvd::tsvd::<f64>;
    let _ = sparse_ir_basis::tsvd::<f64>;

    let m = sparse_ir_core::mat![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(m[[1, 0]], 3.0);
}

#[test]
fn paths_of_the_sve_crate() {
    let _ = sparse_ir::SegmentHints::uniform::<sparse_ir::LogisticKernel>;
    let _ = sparse_ir::hints::SegmentHints::uniform::<sparse_ir::LogisticKernel>;

    let err = sparse_ir::sve::Error::EmptyInput { name: "matrix" };
    assert_eq!(err.kind(), sparse_ir::sve::ErrorKind::InvalidArgument);
}
