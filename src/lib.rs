#![crate_name = "compound_factor_iter"]

#![doc = include_str!("../README.md")]

mod common;

mod ordered_permutation_iter;
pub use ordered_permutation_iter::OrderedPermutationIter;

mod radix_permutation_iter;
pub use radix_permutation_iter::RadixPermutationIter;

mod approx_permutation_iter;
pub use approx_permutation_iter::ApproxPermutationIter;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod letter_distribution;


