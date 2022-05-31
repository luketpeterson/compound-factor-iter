
use std::cmp::Ordering;

use crate::common::*;

/// A fast permutation iterator based on traversal through a mixed-radix space.
/// This is much much faster than the ordered search, although it may return
/// some results out of order.
/// 

pub struct RadixPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    /// See common.rs for explanation
    orderings: Vec<Vec<usize>>,

    /// The current position of the result, as indices into the sorted_letters arrays
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

        let mut permuted_state = vec![0; self.factor_count()];
        for (i, &idx) in self.orderings[ordering_idx].iter().enumerate() {
            permuted_state[idx] = self.state[i];
        }


//FUUUUUCKCKCK GOAT.  We need to fix this several ways...
//1. When we step forward the state, we need to be cognizant of the number of elements for a given factor. 
//2. When we switch to a different regime, we need to premute the state because we can't assume it's
// uniform anymore
//

        //Create an array of factors from the permuted state
        //TODO: We could save a Vec allocation by merging the loops above and below this one
        let mut factors = Vec::with_capacity(self.factor_count());
        for (slot_idx, sorted_idx) in permuted_state.iter().enumerate() {
            factors.push(self.sorted_dists[slot_idx][*sorted_idx].1);
        }

        let result = permuted_state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_letter_idx)| self.sorted_dists[slot_idx][*sorted_letter_idx].0)
            .collect();

        self.execute_combine_fn(&factors)
            .map(|combined_val| (result, combined_val))
    }

    fn step(&mut self) -> (bool, Option<(Vec<usize>, T)>) {

        let factor_count = self.factor_count();

        //TODO, if a component reaches the zero threshold then that is the effective max_digit
        // for that component
        //TODO, if every component is at the zero threshold then we're done iterating
        
        self.state[0] += 1;
        let mut cur_digit = 0;
        while self.state[cur_digit] > self.max_digit {

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
        }

        let mut local_max_digit = 0;
        for &digit in self.state.iter() {
            if digit > local_max_digit {
                local_max_digit = digit;
            }
        }
        if local_max_digit < self.max_digit {
            self.state[0] = self.max_digit;
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
