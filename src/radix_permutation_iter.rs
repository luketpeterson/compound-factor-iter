
use std::cmp::Ordering;

/// A fast permutation iterator based on traversal through a mixed-radix space.
/// This is much much faster than the ordered search, although it may return
/// some results out of order.
/// 

pub struct RadixPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,

    //GOAT, Update this comment to capture the idea that there are multiple ordering regimes
    // that switch as the ordering progresses.
    /// A sorted list of the probabilities of the second-place components, in order to
    ///  hit results that include them sooner in the iteration.
    /// NOTE: There is a tradeoff between the word length and how many components to
    ///  consider.  For example, for a 30-letter word, we're looking at a billion
    ///  results before we've even covered the combinitorics probabilities of the
    ///  most and second most likely letters.  So for above a certain length, we want
    ///  to only consider second place values.  But for a 10-letter word, we'd get
    ///  through the top-two possibilities in only 1000 iterations, so we might want
    ///  to consider a sum of places 2 & 3, or possibly places 2-4.  It all comes
    ///  down the the number of letters vs. the number of iterations that can be
    ///  realistically considered.
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

        let mut orderings = Vec::with_capacity(3);
        //Before we make it to [1, 1, 1, 1, 1, ...]
        orderings.push({
            //NOTE: It seems the best results on the nastiest distributions, i.e. with
            //  the most factors, come from establishing ordering based on the
            //  difference between the top and second places.
            let mut ordering = Vec::with_capacity(factor_count);
            for i in 0..factor_count {                    
                let prob_0 = sorted_dists[i][0].1;    
                let prob_1 = sorted_dists[i][1].1;
    
                ordering.push((prob_0 - prob_1, i));
            }
            ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_a.partial_cmp(&prob_b).unwrap_or(Ordering::Equal));
            let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
            ordering
        });

        //After we pass [1, 1, 1, 1, 1, ...], but
        //Before we make it to [2, 2, 2, 2, 2, ...]
        orderings.push({
            //NOTE: For moderate distributions, the best results come from considering
            // the second-place value
            let mut ordering = Vec::with_capacity(factor_count);
            for i in 0..factor_count {        
                let prob = sorted_dists[i][1].1;
    
                ordering.push((prob, i));
            }
            ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
            let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
            ordering
        });

        //After we pass [2, 2, 2, 2, 2, ...]
        orderings.push({
            //NOTE: For distributions with a small number of factors, it really shouldn't
            // matter much because we can easily iterate the whole set, but we get better
            // results considering the second, third, etc. (up to 3 in this case)
            let mut ordering = Vec::with_capacity(factor_count);
            for i in 0..factor_count {        

                let mut l = 1;
                let mut prob = T::zero();
                while l < factor_count && l < 3 {
                    prob = prob + sorted_dists[i][l].1;
                    l += 1;
                }
    
                ordering.push((prob, i));
            }
            ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
            let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
            ordering
        });

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
            permuted_state[i] = self.state[idx];
        }

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
