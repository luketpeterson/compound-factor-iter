
use std::cmp::Ordering;

/// An expensive permutation iterator guaranteed to return results in order, from highest to lowest.
/// 
/// The algorithm works by exhaustively exploring a "frontier region" where results transition from
/// greater than a previous result to less than a previous result.  Because of the multi-dimensional
/// nature of this frontier, the algorithm must try every possible permutation within the frontier
/// region to be sure the next-highest result is found.
/// 
/// The algorithm can be **insanely** expensive because it needs to invoke the `combination_fn`
/// closure potentially `n*2^n` times at each step, where `n` is the number of factors.
/// Due to the cost, the OrderedPermutationIter is only for situaitions where out-of-order
/// results are unaccaptable.  **Otherwise, [ManhattanPermutationIter](crate::ManhattanPermutationIter) is recommended.**
/// 
/// ## Future Work
///
/// * High performance "Equal" path in OrderedPermutationIter.  Currently OrderedPermutationIter
/// buffers up results with an equal value found during one exploration.  Then it returns
/// results out of that buffer until they are exhausted after which it begins another search.
/// This assumes equal-value results are an exception.  In use-cases where they are numerous, a
/// better approach would be to have two traversal modes, one mode searching for the best
/// next-result and the other mode scanning for the next equal-result.
///
pub struct OrderedPermutationIter<'a, T> {

    /// The individual distributions we're iterating the permutations of
    sorted_dists: Vec<Vec<(usize, T)>>,

    /// A function capable of combining factors
    combination_fn: &'a dyn Fn(&[T]) -> Option<T>,
    
    /// The current position of the result, as indices into the sorted_dists arrays
    state: Vec<usize>,

    /// The highest value that the state has achieved for a given factor
    high_water_mark: Vec<usize>,

    /// The threshold value, corresponding to the last returned result
    current_val: T,

    /// A place to stash future results with values that equal to the last-returned result
    result_stash: Vec<(Vec<usize>, T)>,
}

impl<'a, T> OrderedPermutationIter<'a, T>
    where
    T: Copy + PartialOrd + num_traits::Bounded,
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
            high_water_mark: vec![0; factor_count],
            current_val: T::max_value(),
            result_stash: vec![],
        }
    }
    pub fn factor_count(&self) -> usize {
        self.sorted_dists.len()
    }
    fn factors_from_state(&self, state: &[usize]) -> Vec<T> {

        let mut factors = Vec::with_capacity(state.len());
        for (slot_idx, sorted_idx) in state.iter().enumerate() {
            factors.push(self.sorted_dists[slot_idx][*sorted_idx].1);
        }

        factors
    }
    fn execute_combine_fn(&self, factors: &[T]) -> Option<T> {
        (self.combination_fn)(&factors)
    }
    fn state_to_result(&self) -> Option<(Vec<usize>, T)> {
        
        let result: Vec<usize> = self.state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_factor_idx)| self.sorted_dists[slot_idx][*sorted_factor_idx].0)
            .collect();

        let factors = self.factors_from_state(&self.state);

        self.execute_combine_fn(&factors)
            .map(|combined_val| (result, combined_val))
    }
    /// Searches the frontier around a state, looking for the next state that has the highest overall
    /// combined value, that is lower than the current_val.  Returns None if it's impossible to advance
    /// to a valid value.
    /// 
    fn find_smallest_next_increment(&self) -> Option<Vec<(Vec<usize>, T)>> {

        //Explanation of overall algorithm:
        //
        //The algorithm maintains a "frontier region" between the positions of "tops" and "bottoms"
        //First, "tops" are set, and then "bottoms" are discovered based on "tops".  Higher tops
        // results in less constraining pressure and therfore lower bottoms as well, making the
        // search space explode.  Therefore we need to be very judicious about advancing "tops"
        //
        //This algorithm consists of 3 nested loops.
        //
        //1. The outermost loop is the "factor_to_advance" loop, which iterates for each factor
        // plus a final iteration that advances no factors.  The idea is that we advance each single
        // factor to the furthest point it's ever been to for a new "top" on that factor alone.
        // Then we search for permutations using that particular frontier, and repeat for each factor.
        //
        //2. The "while !finished" loop is the hairy one.  It systematically tries every permutation
        // between tops and bottoms. This loop can iterate 2^n times for n factors, and sometimes more
        //
        //3. The "rollover loop", aka `while temp_state[cur_factor] > tops[cur_factor]` is effectively
        // just an incrementor for a mixed-radix number.  It's carrying forward the increments until
        // it finds a place to put them, or determines the iteration is finished
        //

        let factor_count = self.factor_count();

        let mut highest_val = T::min_value();
        let mut return_val = None;

        //NOTE: when factor_to_advance == factor_count, that means we don't attempt to advance any factor
        for factor_to_advance in 0..(factor_count+1) {

            //The "tops" are the highest values each individual factor could possibly have and still reference
            // the next permutation in the sequence
            let mut skip_factor = false;
            let mut tops = Vec::with_capacity(factor_count);
            for (i , &val) in self.high_water_mark.iter().enumerate() {
                if i == factor_to_advance {
                    if val+1 < self.sorted_dists[i].len() {
                        tops.push(val+1);
                    } else {
                        skip_factor = true;
                    }
                } else {
                    tops.push(val);
                }
            }
            if skip_factor {
                continue;
            }

            //Find the "bottoms", i.e. the lowest value each factor could possibly have given
            // the "tops", without exceeding the threshold established by self.current_val
            let mut bottoms = Vec::with_capacity(factor_count);
            for i in 0..factor_count {
                let old_top = tops[i];
                let mut new_bottom = self.state[i];
                loop {
                    if new_bottom == 0 {
                        bottoms.push(0);
                        break;
                    }
                    tops[i] = new_bottom; //Temporarily hijacking tops
                    let factors = self.factors_from_state(&tops);
                    let val = self.execute_combine_fn(&factors);
                    if val.is_some() && val.unwrap() > self.current_val {
                        bottoms.push(new_bottom+1);
                        break;
                    } else {
                        new_bottom -= 1;
                    }
                }
                tops[i] = old_top;
            }

            //We need to check every combination of adjustments between tops and bottoms
            let mut temp_state = bottoms.clone();
            let mut temp_factors = self.factors_from_state(&temp_state);
            if factor_to_advance < factor_count {
                temp_state[factor_to_advance] = tops[factor_to_advance];
                temp_factors[factor_to_advance] = self.sorted_dists[factor_to_advance][temp_state[factor_to_advance]].1;
            }
            let mut finished = false;
            while !finished {
    
                //Increment the adjustments to the next state we want to try
                //NOTE: It is impossible for the initial starting case (all bottoms) to be the
                // next sequence element, because it's going to be the current sequence element
                // or something earlier
                let mut cur_factor;
                if factor_to_advance != 0 {
                    temp_state[0] += 1;
                    if temp_state[0] < self.sorted_dists[0].len() {
                        temp_factors[0] = self.sorted_dists[0][temp_state[0]].1;
                    }
                    cur_factor = 0;
                } else {
                    temp_state[1] += 1;
                    if temp_state[1] < self.sorted_dists[1].len() {
                        temp_factors[1] = self.sorted_dists[1][temp_state[1]].1;
                    }
                    cur_factor = 1;
                }

                //Deal with any rollover caused by the increment above
                while temp_state[cur_factor] > tops[cur_factor] {

                    temp_state[cur_factor] = bottoms[cur_factor];
                    temp_factors[cur_factor] = self.sorted_dists[cur_factor][temp_state[cur_factor]].1;
                    cur_factor += 1;

                    //Skip over the factor_to_advance, which we're going to leave pegged to tops
                    if cur_factor == factor_to_advance {
                        cur_factor += 1;
                    }

                    if cur_factor < factor_count {
                        temp_state[cur_factor] += 1;
                        if temp_state[cur_factor] < self.sorted_dists[cur_factor].len() {
                            temp_factors[cur_factor] = self.sorted_dists[cur_factor][temp_state[cur_factor]].1;
                        }
                    } else {
                        finished = true;
                        break;
                    }
                }
    
                if let Some(temp_val) = self.execute_combine_fn(&temp_factors) {
                    if temp_val < self.current_val && temp_val >= highest_val {

                        if temp_val > highest_val {
                            //Replace the results with a fresh array
                            highest_val = temp_val;
                            return_val = Some(vec![(temp_state.clone(), highest_val)]);
                        } else {
                            //We can infer temp_val == highest_val if we got here, so
                            // append to the results array
                            return_val.as_mut().unwrap().push((temp_state.clone(), highest_val));
                        }
                    }
                }
            }
        }

        //See if there are any additional results with the same combined value, adjacent to the
        // results we found
        if let Some(results) = &mut return_val.as_mut() {
            let mut new_results = results.clone();
            for (result, val) in results.iter() {
                self.find_adjacent_equal_permutations(result, *val, &mut new_results);
            }
            **results = new_results;
        }

        return_val
    }
    //An Adjacent Permutation is defined as a permutation that can be created by adding 1 to one
    // factor.  This function will find all adjacent permutations from the supplied state, with a
    // value equal to the supplied "val" argument.  It will also find the equal permutations from
    // all found permutations, recursively.
    fn find_adjacent_equal_permutations(&self, state: &[usize], val: T, results: &mut Vec<(Vec<usize>, T)>) {

        let factor_count = self.factor_count();
        let mut new_state = state.to_owned();
        
        loop {

            //Increment the state by 1 and get the new value
            new_state[0] += 1;
            let mut cur_digit = 0;
            let mut temp_val = if new_state[cur_digit] < self.sorted_dists[cur_digit].len() {
                let factors = self.factors_from_state(&new_state);
                self.execute_combine_fn(&factors)
            } else {
                None
            };

            //Deal with the rollover caused by the previous increment
            //NOTE: This loop has two continuing criteria. 1.) If we must roll over because
            // we've incremented one factor to the end, and 2.) If the new combined value is too
            // small, indicating the factor shouldn't be considered in an equal permutation
            while new_state[cur_digit] == self.sorted_dists[cur_digit].len()
                || (temp_val.is_some() && temp_val.unwrap() < val) {

                new_state[cur_digit] = state[cur_digit];
                cur_digit += 1;

                if cur_digit == factor_count {
                    break;
                }

                new_state[cur_digit] += 1;
                if new_state[cur_digit] < self.sorted_dists[cur_digit].len() {
                    let factors = self.factors_from_state(&new_state);
                    temp_val = self.execute_combine_fn(&factors);
                }
            }
            
            if temp_val.is_some() && temp_val.unwrap() == val {
                //Check for duplicates, and add this state if it's unique
                if results.iter().position(|(element_state, _val)| *element_state == new_state).is_none() {
                    results.push((new_state.clone(), val));
                }
            } else {
                break;
            }
        }
    }
}

impl<T> Iterator for OrderedPermutationIter<'_, T>
    where
    T: Copy + PartialOrd + num_traits::Bounded,
{
    type Item = (Vec<usize>, T);

    fn next(&mut self) -> Option<Self::Item> {
        
        let factor_count = self.factor_count();

        //If we have some results in the stash, return those first
        if let Some((new_state, new_val)) = self.result_stash.pop() {
            self.state = new_state;
            self.current_val = new_val;

            return self.state_to_result();
        }

        //Find the next configuration with the smallest incremental impact to the combined value
        if let Some(new_states) = self.find_smallest_next_increment() {
        
            //Advance the high-water mark for all returned states
            for (new_state, _new_val) in new_states.iter() {
                for i in 0..factor_count {
                    if new_state[i] > self.high_water_mark[i] {
                        self.high_water_mark[i] = new_state[i];
                    }
                }
            }

            //Stash all the results we got
            self.result_stash = new_states;

            //Return one result from our stash
            let (new_state, new_val) = self.result_stash.pop().unwrap();
            self.state = new_state;
            self.current_val = new_val;

            return self.state_to_result();
                
        } else {
            //If we couldn't find any factors to advancee, we've reached the end of the iteration
            return None;
        }
    }
}
