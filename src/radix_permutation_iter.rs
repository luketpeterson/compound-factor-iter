
use std::cmp::Ordering;

/// A fast permutation iterator based on counting with a mixed-radix number.
/// 
/// If you think of each factor as a digit in a number, this iterator will "count" through all
/// possible permutations, but it caps each digit's base (aka radix).  When it's impossible
/// to increment the counter, it scans all factors (digits) to find the best cap to increase,
/// and then continues counting with the increased cap.
///
/// The RadixPermutationIter produces a reasonably high quality sequence, but can create some
/// outlier errors, when many digits "roll over".
/// 
/// The [ManhattanPermutationIter](crate::ManhattanPermutationIter) will usually produce more
/// uniform coverage of a the factor-space, while the RadixPermutationIter will offer a more
/// orderly sequence over small intervals.
/// 
/// RadixPermutationIter is appropriate when some factors have vastly more influence over the
/// `combination_fn` function's result than others.  This can be caused by the function itself
/// or by the distribution of the input factor values.  It is recommended to try both iterators
/// to determine the best one for your situation.
/// 
/// ## Sequence Characterisitcs
/// 
/// On average, all results in the set of `n` iterations of the [OrderedPermutationIter](crate::OrderedPermutationIter) will occur
/// in the set of `k*n^2` iterations of the RadixPermutationIter, where `k` is a constant.
/// Empirically I have found `k` is usually between `0.1` and `0.01` for the data sets I've tested.
/// 
/// For very long sequences with uniform factor influence, RadixPermutationIter is somewhat
/// worse than [ManhattanPermutationIter](crate::ManhattanPermutationIter),
/// however there are certain situations where RadixPermutationIter will produce a better sequence,
/// especially with shorter sequences.
/// 
pub struct RadixPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    /// The current position of the result, as indices into the sorted_dists arrays
    state: Vec<usize>,

    /// The maximum digit position for each factor.  This allows us to increase the radix
    /// of each factor individually
    max_factors: Vec<usize>,

    /// The maximum value for the digit with the most elements
    global_max_digit: usize,

    /// The index of the digit that's pegged to its maximum value, because we just incremented it
    pegged_factor: usize,

    /// Counts the internal steps because initialization is a degenerate case
    /// NOTE: this doesn't correspond to returned results because we don't return results when the
    /// combination_fn returns `None`.
    step_count: usize,
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

        Self {
            sorted_dists,
            combination_fn,
            state: vec![0; factor_count],
            max_factors: vec![0; factor_count],
            global_max_digit,
            pegged_factor: 0,
            step_count: 0,
        }
    }
    pub fn factor_count(&self) -> usize {
        self.sorted_dists.len()
    }
    fn execute_combine_fn(&self, factors: &[T]) -> Option<T> {
        (self.combination_fn)(&factors)
    }
    fn state_to_result(&self) -> Option<(Vec<usize>, T)> {

        //Create an array of factors to call the combination_fn
        let mut factors = Vec::with_capacity(self.factor_count());
        for (slot_idx, sorted_idx) in self.state.iter().enumerate() {
            factors.push(self.sorted_dists[slot_idx][*sorted_idx].1);
        }

        let result = self.state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_factor_idx)| self.sorted_dists[slot_idx][*sorted_factor_idx].0)
            .collect();

        self.execute_combine_fn(&factors)
            .map(|combined_val| (result, combined_val))
    }

    fn step(&mut self) -> (bool, Option<(Vec<usize>, T)>) {

        let factor_count = self.factor_count();

        //Increment the first non-pegged digit
        let mut cur_digit = if self.pegged_factor != 0 {
            0
        } else {
            1
        };
        self.state[cur_digit] += 1;

        //Roll over, if necessary
        while self.state[cur_digit] > self.max_factors[cur_digit] ||
            self.state[cur_digit] >= self.sorted_dists[cur_digit].len() {

            self.state[cur_digit] = 0;
            cur_digit += 1;

            //Skip over the pegged factor
            if cur_digit == self.pegged_factor {
                cur_digit += 1;
            }

            //See if we've found a digit to roll into
            if cur_digit < factor_count {
                self.state[cur_digit] += 1;
            } else {
                //If we came to the end of the number and didn't find a place to roll into,
                // then we need to increase the radix of one digit

                //Find the factor to advance, and move it forward
                if let Some(factor_to_advance) = self.find_factor_to_advance() {
                    self.max_factors[factor_to_advance] += 1;
                    self.state[self.pegged_factor] = 0;
                    self.pegged_factor = factor_to_advance;
                    self.state[factor_to_advance] = self.max_factors[factor_to_advance];
                    if factor_to_advance != 0 {
                        cur_digit = 0;
                    } else {
                        cur_digit = 1;
                    }
                } else {
                    //We've finished the iteration...
                    return (false, None);
                }
            }
        }

        (true, self.state_to_result())
    }

    fn find_factor_to_advance(&self) -> Option<usize> {

        let mut best_factor: Option<(usize, T)> = None;
        for (factor_idx, &factor_pos) in self.max_factors.iter().enumerate() {
            if factor_pos < self.global_max_digit && 
                factor_pos < self.sorted_dists[factor_idx].len()-1 {

                let mut factors: Vec<T> = self.sorted_dists.iter().map(|inner_dist| inner_dist[0].1).collect();
                factors[factor_idx] = self.sorted_dists[factor_idx][factor_pos+1].1;
                let comb_val = (self.combination_fn)(&factors).unwrap_or(T::zero());
    
                if let Some((best_idx, best_val)) = best_factor.as_mut() {
                    if comb_val > *best_val {
                        *best_idx = factor_idx;
                        *best_val = comb_val;
                    }
                } else {
                    best_factor = Some((factor_idx, comb_val));
                }
            }
        }

        best_factor.map(|(best_idx, _best_val)| best_idx)
    }
}

impl<T> Iterator for RadixPermutationIter<'_, T>
    where
    T: Copy + PartialOrd + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>,
{
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {

        match self.step_count {
            0 => {
                self.step_count = 1;
                return self.state_to_result();
            },
            1 => {
                self.pegged_factor = self.find_factor_to_advance().unwrap();
                self.max_factors[self.pegged_factor] = 1;
                self.state[self.pegged_factor] = 1;
                self.step_count = 2;
                return self.state_to_result();
            },
            _ => {
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
    }
}
