#![crate_name = "compound_factor_iter"]

#![doc = include_str!("../README.md")]

mod common;

mod ordered_permutation_iter;
pub use ordered_permutation_iter::OrderedPermutationIter;

mod radix_permutation_iter;
pub use radix_permutation_iter::RadixPermutationIter;

mod manhattan_permutation_iter;
pub use manhattan_permutation_iter::ManhattanPermutationIter;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod letter_distribution;


