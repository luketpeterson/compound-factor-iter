
use std::cmp::Ordering;

use crate::common::*;

/// A fast permutation iterator based on conserving path distance in an n-dimensional hypercube
/// 
/// ManhattanPermutationIter is **much much** faster than [OrderedPermutationIter](crate::OrderedPermutationIter), although it
/// may return some results out of order.
/// 
/// The word "Manhattan" comes from the fact that permutations are tried in an
/// order determined by their [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry)
/// from the single best permutation.
/// 
/// If you conceptualize the space of all possible permutations as an n-dimensional hypercube with
/// one dimension for each factor, and each step along a dimension results in the swapping of a
/// factor's value for the next-smaller value.  The ManhattanPermutationIter will systematically
/// explore all permutations at each distance before incrementing the distance by one.
/// 
/// ## Sequence Characterisitcs
/// 
/// On average, all results in the set of `n` iterations of the [OrderedPermutationIter](crate::OrderedPermutationIter) will occur
/// in the set of `k*n*log(n)` iterations of the ManhattanPermutationIter, where `k` is a constant
/// that is affected by the distribution of input factors and the `combination_fn`.  Empirically I
/// have found `k` is usually between `10` and `50` for the data sets I've tested.
/// 
/// ## Algorithm Details
/// 
/// The algorithm begins with a search distance of 0, and therefore returns the the best permutation.
/// Each subsequent iteration either shifts the factors to find another permutation with the same
/// manhattan search distance or it increases the search distance by 1.
/// 
/// When the search distance is incremented, the the algorithm starts at the most significant factor,
/// and apportions all of the "distance budget" to that factor.  On each subsequent iteration,
/// some of the distance budget is shifted to less signifigant factors.
///
/// For example, if we are searching at a total distance of 2, the iterations would look like:
/// ```txt
/// 2, 0, 0
/// 1, 1, 0
/// 1, 0, 1
/// 0, 2, 0
/// 0, 1, 1
/// 0, 0, 2
/// ```
/// 
pub struct ManhattanPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    /// See common.rs for explanation
    orderings: Vec<Vec<usize>>,

    /// The current position of the result, as indices into the sorted_dists arrays
    state: Vec<usize>,

    /// The manhattan distance of the current search
    distance_threshold: usize,

    /// Tracks whether we need to expand the distance_threshold on the next iteration
    expand_distance_threshold: bool,

    /// Initialization is a degenerate case
    new_iter: bool,
}

impl<'a, T> ManhattanPermutationIter<'a, T> 
    where
    T: Copy + PartialOrd + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>,
{
    pub fn new<E: AsRef<[T]>, F: Fn(&[T]) -> Option<T>>(factor_iter: impl Iterator<Item=E>, combination_fn: &'a F) -> Self {

        let sorted_dists: Vec<Vec<(usize, T)>> = factor_iter
            .map(|factor_dist| {
                let mut sorted_elements: Vec<(usize, T)> = factor_dist.as_ref().iter().cloned().enumerate().collect();
                sorted_elements.sort_by(|(_idx_a, element_a), (_idx_b, element_b)| element_b.partial_cmp(element_a).unwrap_or(Ordering::Equal));                
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
            distance_threshold: 0,
            expand_distance_threshold: false,
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
        let ordering_idx = if self.distance_threshold > 0 {
            (self.distance_threshold-1).min(2)
        } else {
            0
        };

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
        let ordering_idx = if self.distance_threshold > 0 {
            (self.distance_threshold-1).min(2)
        } else {
            0
        };

        let factor_count = self.factor_count();

        //TODO, if a factor reaches the zero threshold then that is the effective max_digit
        // for that factor
        //TODO, if every component is at the zero threshold then we're done iterating
        //Update: I'm not sure we can rely on that property given arbitrary types and arbitrary
        // combination functions

        // Scan the state from right to left looking for the first value that can be shifted right.
        let mut found_factor = factor_count-1;
        let mut swizzled_factor = self.orderings[ordering_idx][found_factor];
        while found_factor > 0 && (self.state[found_factor-1] == 0 || self.state[found_factor] == self.sorted_dists[swizzled_factor].len()-1) {
            found_factor -= 1;
            swizzled_factor = self.orderings[ordering_idx][found_factor];
        }

        //Make sure we have something to decrement and that we also have a place to put that value
        if found_factor == 0 || self.expand_distance_threshold {

            self.expand_distance_threshold = false;

            //If we can't do the decrement then increase the distance_threshold
            self.distance_threshold += 1;

            //Reset the state for scanning permutations at the new distance_threshold
            let mut remaining_distance = self.distance_threshold;
            for (i, factor) in self.state.iter_mut().enumerate() {
                let swizzled_i = self.orderings[ordering_idx][i];
                let max_factor = self.sorted_dists[swizzled_i].len()-1;
                if remaining_distance > max_factor {
                    *factor = max_factor;
                    remaining_distance -= max_factor;
                } else {
                    *factor = remaining_distance;
                    remaining_distance = 0;
                }
            }

            //If we still have some distance left over then we're done iterating
            if remaining_distance > 0 {
                return (false, None);
            }

            // println!("DebugPrint jump_thresh to {}, {:?}", self.distance_threshold, self.state);
        } else {

            // Decrement the value we found by 1
            self.state[found_factor-1] -= 1;

            // Figure out how much "distance" lies to the left of the factor we found.  distance_threshold
            // minus the allocated distance to the left is the remaining_distance we have to work with.
            let mut allocated_distance = 0;
            for i in 0..(found_factor) {
                allocated_distance += self.state[i];
            }
            let mut remaining_distance = self.distance_threshold - allocated_distance;

            // Put the remaining "distance" into the factor(s) immediately to the right of that
            // decremented factor, and zero out the remainder of the factors to the right.
            for i in found_factor..factor_count {
                let swizzled_i = self.orderings[ordering_idx][i];
                let max_factor = self.sorted_dists[swizzled_i].len()-1;
                if remaining_distance > max_factor {
                    self.state[i] = max_factor;
                    remaining_distance -= max_factor;
                } else {
                    self.state[i] = remaining_distance;
                    remaining_distance = 0;
                }
            }

            //If we still have some distance remaining then we need to expand the distance_threshold
            if remaining_distance > 0 {
                // println!("DebugPrint EXPAND ff= {}, remd={}, {:?}", found_factor, remaining_distance, self.state);
                self.expand_distance_threshold = true;
                return (true, None);
            }

            // println!("DebugPrint increm ff= {}, remd={}, {:?}", found_factor, remaining_distance, self.state);
        }

        (true, self.state_to_result())
    }
}

impl<T> Iterator for ManhattanPermutationIter<'_, T>
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

//More implementation examples
//
//state = 2, 0, 0   starting state,
//state = 1, 1, 0   decrement 0th, put ALL remaining into 1st, and zero-out digits to the right,
//state = 1, 0, 1   decrement 1st, put ALL remaining into 2nd, and zero-out digits to the right,
//state = 0, 2, 0   decrement 0th, put ALL remaining into 1st, and zero-out digits to the right,

//Example with 3 factors, and a distance_threshold of 3
//- 3, 0, 0
//- 2, 1, 0
//- 2, 0, 1
//- 1, 2, 0
//- 1, 1, 1
//- 1, 0, 2
//- 0, 3, 0
//- 0, 2, 1
//- 0, 1, 2
//- 0, 0, 3
