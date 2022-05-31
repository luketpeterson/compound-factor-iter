
use std::cmp::Ordering;

/// A fast permutation iterator based on traversal through a mixed-radix space.
/// This is much much faster than the ordered search, although it may return
/// some results out of order.
/// 

pub struct ApproxPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    /// The current position of the result, as indices into the sorted_letters arrays
    state: Vec<usize>,

    /// The manhattan distance of the current search
    distance_threshold: usize,

    /// Tracks whether we need to expand the distance_threshold on the next iteration
    expand_distance_threshold: bool,

    /// Initialization is a degenerate case
    new_iter: bool,
}

impl<'a, T> ApproxPermutationIter<'a, T> 
    where
    T: Copy + PartialOrd// + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>, GOAT
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

        Self {
            sorted_dists,
            combination_fn,
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

        //GOAT, we want to actually do the permute
        let permuted_state = self.state.clone();

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

        // Scan the state from right to left looking for the first value that can be shifted right.
        let mut found_factor = factor_count-1;
        while found_factor > 0 && self.state[found_factor-1] == 0 {
            found_factor -= 1;
        }

        // // Add up the factors to the right of the place we found, and make sure we can receive the
        // // new value
        // let mut distance_to_right = 0;
        // for i in found_factor..factor_count {
        //     distance_to_right += self.state[i];
        // }

        //GOAT, this "distance_to_right >= self.distance_threshold" check is obv wrong.  The rest I'm feeling good about.
        // although I also need to think through whether overflowing some factors to their neighbors is acceptable, or whether it
        // will lead to duplicated results...

        //Make sure we have something to decrement and that we also have a place to put that value
        if (found_factor == 0 && self.state[0] == 0) || self.expand_distance_threshold {

            self.expand_distance_threshold = false;

            //If we can't do the decrement then increase the distance_threshold
            self.distance_threshold += 1;

            //Reset the state for scanning permutations at the new distance_threshold
            let mut remaining_distance = self.distance_threshold;
            for (i, factor) in self.state.iter_mut().enumerate() {
                let max_factor = self.sorted_dists[i].len()-1;
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

            println!("GOAT jump_thresh to {}, {:?}", self.distance_threshold, self.state);
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

            //GOAT dead
            // // Zero-out all of the remaining factors to the right of the factor we decremented
            // // NOTE: Conceptually we zero our the remaining digits, but we just scanned them and
            // // verified they are zero, so in practice, we can just set the last digit.
            // *self.state.last_mut().unwrap() = 0;

            // Put the remaining "distance" into the factor(s) immediately to the right of that
            // decremented factor, and zero out the remainder of the factors to the right.
            for i in found_factor..factor_count {
                let max_factor = self.sorted_dists[i].len()-1;
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
                //println!("GOAT EXPAND ff= {}, remd={}, {:?}", found_factor, remaining_distance, self.state);
                self.expand_distance_threshold = true;
                return (true, None);
            }


            println!("GOAT increm ff= {}, remd={}, {:?}", found_factor, remaining_distance, self.state);
        }




        // self.state[0] += 1;
        // let mut cur_digit = 0;
        // while self.state[cur_digit] > self.max_digit {

        //     self.state[cur_digit] = 0;
        //     cur_digit += 1;

        //     if cur_digit < factor_count {
        //         self.state[cur_digit] += 1;
        //     } else {
        //         if self.max_digit < self.global_max_digit {
        //             self.max_digit += 1;
        //             cur_digit = 0;
        //         } else {
        //             //We've finished the iteration...
        //             return (false, None);
        //         }
        //     }
        // }

        // let mut local_max_digit = 0;
        // for &digit in self.state.iter() {
        //     if digit > local_max_digit {
        //         local_max_digit = digit;
        //     }
        // }
        // if local_max_digit < self.max_digit {
        //     self.state[0] = self.max_digit;
        // }

        (true, self.state_to_result())
    }
}

impl<T> Iterator for ApproxPermutationIter<'_, T>
    where
    T: Copy + PartialOrd// + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>, GOAT
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

//Improvement upon the Radix Iterator: I have a suspicion that a better algorithm exists that is able to
// systematically traverse the factor-space while causing the sum-of-sorted-factor-places to
// increase monotonically, and not miss any permutations.  This would get rid of the largest
// source of out-of-order results for the radix iterator, which is the rollover behavior,
// where incrementing one factor resets many factors back to a much lower position.
//
//The "Manhattan" Iterator would conserve manhattan distance.  That is, each solution would be the same
// or more "steps" away from the best solution.  The implementation would monotonically increment a
// distance, and distribute that distance based on a state.  Only permutations that allowed for all of the
// distance to be distributed exactly would be accepted.
//
//So, for example, if we need to distribute a total distance of 2, the iterations might look like:
//- 2, 0, 0
//- 1, 1, 0
//- 1, 0, 1
//- 0, 2, 0
//- 0, 1, 1
//- 0, 0, 2
//
//
//
//The algorithm starts at the most significant place, and apportions an ammount based on a state that
// iterates from "remaining_distance_budget" to 0.  then, the remaining budget is given to the next
// place, and on it goes with the state for that place iterating from remaining budget to 0.
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
