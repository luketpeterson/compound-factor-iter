
use std::cmp::Ordering;

use crate::common::*;

/// A fast permutation iterator based on a counting with a mixed-radix number.
/// 
/// In the majority of situations, the [ManhattanPermutationIter](crate::ManhattanPermutationIter)
/// will produce a higher quality sequence, but the RadixPermutationIter is appropriate when
/// some factors have vastly more influence over the function result than others.  This can
/// be caused by the function itself or by the input factor values, so it is recommended to
/// try both iterators to determine the best one for your situation.
/// 
/// ## Sequence Characterisitcs
/// 
/// On average, all results in the set of `n` iterations of the [OrderedPermutationIter](crate::OrderedPermutationIter) will occur
/// in the set of `k*n^2` iterations of the RadixPermutationIter, where `k` is a constant.
/// Empirically I have found `k` is usually between `3` and `10` for the data sets I've tested.
/// 
/// For long sequences, RadixPermutationIter is considerably worse than [ManhattanPermutationIter](crate::ManhattanPermutationIter),
/// however there are certain situations where RadixPermutationIter will produce a better sequence,
/// especially with short sequences and within certain bounded ranges of results.
/// 
pub struct RadixPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    /// See common.rs for explanation
    orderings: Vec<Vec<usize>>,

    /// The current position of the result, as indices into the sorted_dists arrays
    state: Vec<usize>,

    /// The maximum digit position across the entire state vec.  This allows us to evenly
    /// increment the low positions of every digit before attempting to set any digits to
    /// relatively high (improbable) values.
    max_digit: usize,

    /// The maximum value for the digit with the most elements
    global_max_digit: usize,

    /// Initialization is a degenerate case
    new_iter: bool,
}

impl<'a, T> RadixPermutationIter<'a, T> 
    where
    T: Copy + PartialOrd + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>,
{
    pub fn new<E: AsRef<[T]>, F: Fn(&[T]) -> Option<T>>(factor_iter: impl Iterator<Item=E>, combination_fn: &'a F) -> Self {

        let mut global_max_digit = 0;
        let sorted_dists: Vec<Vec<(usize, T)>> = factor_iter
            .map(|factor_dist| {
                let mut sorted_elements: Vec<(usize, T)> = factor_dist.as_ref().iter().cloned().enumerate().collect();
                sorted_elements.sort_by(|(_idx_a, element_a), (_idx_b, element_b)| element_b.partial_cmp(element_a).unwrap_or(Ordering::Equal));
                
                if sorted_elements.len()-1 > global_max_digit {
                    global_max_digit = sorted_elements.len()-1;
                }
                sorted_elements
            })
            .collect();

        let factor_count = sorted_dists.len();

        let orderings = build_orderings(&sorted_dists, &combination_fn);

        Self {
            sorted_dists,
            combination_fn,
            orderings,
            state: vec![0; factor_count],
            max_digit: 1,
            global_max_digit,
            new_iter: true,
        }
    }
    pub fn factor_count(&self) -> usize {
        self.sorted_dists.len()
    }
    fn execute_combine_fn(&self, factors: &[T]) -> Option<T> {
        (self.combination_fn)(&factors)
    }
    fn state_to_result(&self) -> Option<(Vec<usize>, T)> {

        //Which ordering we use depends on how far into the sequence we are
        let ordering_idx = (self.max_digit-1).min(2);

        let mut swizzled_state = vec![0; self.factor_count()];
        for (i, &idx) in self.orderings[ordering_idx].iter().enumerate() {
            swizzled_state[idx] = self.state[i];
        }

        //Create an array of factors from the swizzled state
        //TODO: We could save a Vec allocation by merging the loops above and below this one
        let mut factors = Vec::with_capacity(self.factor_count());
        for (slot_idx, sorted_idx) in swizzled_state.iter().enumerate() {
            factors.push(self.sorted_dists[slot_idx][*sorted_idx].1);
        }

        let result = swizzled_state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_factor_idx)| self.sorted_dists[slot_idx][*sorted_factor_idx].0)
            .collect();

        self.execute_combine_fn(&factors)
            .map(|combined_val| (result, combined_val))
    }

    fn step(&mut self) -> (bool, Option<(Vec<usize>, T)>) {

        //Which ordering we use depends on how far into the sequence we are
        let ordering_idx = (self.max_digit-1).min(2);

        let factor_count = self.factor_count();

        //TODO, if a factor reaches the zero threshold then that is the effective max_digit
        // for that factor
        //TODO, if every component is at the zero threshold then we're done iterating
        //Update: I'm not sure we can rely on that property given arbitrary types and arbitrary
        // combination functions

        self.state[0] += 1;
        let mut cur_digit = 0;
        let mut swizzled_cur_digit = self.orderings[ordering_idx][cur_digit];
        while self.state[cur_digit] > self.max_digit ||
            self.state[cur_digit] >= self.sorted_dists[swizzled_cur_digit].len() {

            self.state[cur_digit] = 0;
            cur_digit += 1;

            if cur_digit < factor_count {
                self.state[cur_digit] += 1;
            } else {
                if self.max_digit < self.global_max_digit {
                    self.max_digit += 1;
                    cur_digit = 0;
                } else {
                    //We've finished the iteration...
                    return (false, None);
                }
            }

            swizzled_cur_digit = self.orderings[ordering_idx][cur_digit];
        }

        //If all digits are below self.max_digit, then we've already hit this permutations when
        // self.max_digit had a lower value, so step forward to a state we definitely haven't
        // visited yet.
        let mut local_max_digit = 0;
        for &digit in self.state.iter() {
            if digit > local_max_digit {
                local_max_digit = digit;
            }
        }
        if local_max_digit < self.max_digit {
            let mut i = 0;
            let mut swizzled_i = self.orderings[ordering_idx][i];
            while self.max_digit >= self.sorted_dists[swizzled_i].len() {
                self.state[i] = 0;
                i += 1;
                swizzled_i = self.orderings[ordering_idx][i];
            }
            self.state[i] = self.max_digit;
        }

        (true, self.state_to_result())
    }
}

impl<T> Iterator for RadixPermutationIter<'_, T>
    where
    T: Copy + PartialOrd + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>,
{
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {

        if self.new_iter {
            self.new_iter = false;
            return self.state_to_result();
        }

        //We don't want to stop iterating until we've actually exhausted all permutations,
        // since we can't guarantee there aren't better options to come
        loop {
            let (keep_going, result_option) = self.step();
            if !keep_going {
                return None;
            }
            if result_option.is_some() {
                return result_option;
            }
        }
    }
}
