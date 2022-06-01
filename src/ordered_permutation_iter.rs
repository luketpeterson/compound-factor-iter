
use std::cmp::Ordering;

///GOAT, better description, explain algorithm
/// 
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
    
    /// The current position of the result, as indices into the sorted_letters arrays
    state: Vec<usize>,

    /// The highest value state has achieved for a given letter
    high_water_mark: Vec<usize>,

    /// The threshold probability, corresponding to the last returned result
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
            .map(|(slot_idx, sorted_letter_idx)| self.sorted_dists[slot_idx][*sorted_letter_idx].0)
            .collect();

        let factors = self.factors_from_state(&self.state);

        self.execute_combine_fn(&factors)
            .map(|combined_val| (result, combined_val))
    }
    /// Searches the frontier around a state, looking for the next state that has the highest overall
    /// probability, that is lower than the prob_threshold.  Returns None if it's impossible to advance
    /// to a non-zero probability.
    /// 
    fn find_smallest_next_increment(&self) -> Option<Vec<(Vec<usize>, T)>> {

        let factor_count = self.factor_count();

        let mut highest_prob = T::min_value();
        let mut return_val = None;

        //GOAT TODO: Write up a better explanation of the alforithm overall, once it's fully debugged

        //NOTE: when letter_to_advance == factor_count, that means we don't attempt to advance any letter
        for letter_to_advance in 0..(factor_count+1) {

            //The "tops" are the highest values each individual letter could possibly have and still reference
            // the next combination in the sequence
            let mut skip_letter = false;
            let mut tops = Vec::with_capacity(factor_count);
            for (i , &val) in self.high_water_mark.iter().enumerate() {
                if i == letter_to_advance {
                    if val+1 < self.sorted_dists[i].len() {
                        tops.push(val+1);
                    } else {
                        skip_letter = true;
                    }
                } else {
                    tops.push(val);
                }
            }
            if skip_letter {
                continue;
            }

            //Find the "bottoms", i.e. the lowest value each letter could possibly have
            // given the "tops", without exceeding the probability threshold established by
            // self.current_prob
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
                    let prob = self.execute_combine_fn(&factors);
                    if prob.is_some() && prob.unwrap() > self.current_val {
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
            if letter_to_advance < factor_count {
                temp_state[letter_to_advance] = tops[letter_to_advance];
                temp_factors[letter_to_advance] = self.sorted_dists[letter_to_advance][temp_state[letter_to_advance]].1;
            }
            let mut finished = false;
            while !finished {
    
                //Increment the adjustments to the next state we want to try
                //NOTE: It is impossible for the initial starting case (all bottoms) to be the
                // next sequence element, because it's going to be the current sequence element
                // or something earlier
                let mut cur_letter;
                if letter_to_advance != 0 {
                    temp_state[0] += 1;
                    if temp_state[0] < self.sorted_dists[0].len() {
                        temp_factors[0] = self.sorted_dists[0][temp_state[0]].1;
                    }
                    cur_letter = 0;
                } else {
                    temp_state[1] += 1;
                    if temp_state[1] < self.sorted_dists[1].len() {
                        temp_factors[1] = self.sorted_dists[1][temp_state[1]].1;
                    }
                    cur_letter = 1;
                }

                //Deal with any rollover caused by the increment above
                while temp_state[cur_letter] > tops[cur_letter] {

                    temp_state[cur_letter] = bottoms[cur_letter];
                    temp_factors[cur_letter] = self.sorted_dists[cur_letter][temp_state[cur_letter]].1;
                    cur_letter += 1;

                    //Skip over the letter_to_advance, which we're going to leave pegged to tops
                    if cur_letter == letter_to_advance {
                        cur_letter += 1;
                    }

                    if cur_letter < factor_count {
                        temp_state[cur_letter] += 1;
                        if temp_state[cur_letter] < self.sorted_dists[cur_letter].len() {
                            temp_factors[cur_letter] = self.sorted_dists[cur_letter][temp_state[cur_letter]].1;
                        }
                    } else {
                        finished = true;
                        break;
                    }
                }
    
                if let Some(temp_prob) = self.execute_combine_fn(&temp_factors) {
                    if temp_prob < self.current_val && temp_prob >= highest_prob {

                        if temp_prob > highest_prob {
                            //Replace the results with a fresh array
                            highest_prob = temp_prob;
                            return_val = Some(vec![(temp_state.clone(), highest_prob)]);
                        } else {
                            //We can infer temp_prob == highest_prob if we got here, so
                            // append to the results array
                            return_val.as_mut().unwrap().push((temp_state.clone(), highest_prob));
                        }
                    }
                }
            }
        }

        //See if there are any additional results with the same probability, adjacent to the
        // results we found
        if let Some(results) = &mut return_val.as_mut() {
            let mut new_results = results.clone();
            for (result, prob) in results.iter() {
                self.find_adjacent_equal_permutations(result, *prob, &mut new_results);
            }
            **results = new_results;
        }

        return_val
    }
    //An Adjacent Permutation is defined as a permutation that can be created by adding 1 to one
    // factor.  This function will find all adjacent permutations from the supplied state, with a
    // value equal to the supplied "prob" argument.  It will also find the equal permutations from
    // all found permutations, recursively.
    fn find_adjacent_equal_permutations(&self, state: &[usize], prob: T, results: &mut Vec<(Vec<usize>, T)>) {

        let letter_count = self.factor_count();
        let mut new_state = state.to_owned();
        
        loop {

            //Increment the state by 1 and get the new value
            new_state[0] += 1;
            let mut cur_digit = 0;
            let mut temp_prob = if new_state[cur_digit] < self.sorted_dists[cur_digit].len() {
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
                || (temp_prob.is_some() && temp_prob.unwrap() < prob) {

                new_state[cur_digit] = state[cur_digit];
                cur_digit += 1;

                if cur_digit == letter_count {
                    break;
                }

                new_state[cur_digit] += 1;
                if new_state[cur_digit] < self.sorted_dists[cur_digit].len() {
                    let factors = self.factors_from_state(&new_state);
                    temp_prob = self.execute_combine_fn(&factors);
                }
            }
            
            if temp_prob.is_some() && temp_prob.unwrap() == prob {
                //Check for duplicates, and add this state if it's unique
                if results.iter().position(|(element_state, _prob)| *element_state == new_state).is_none() {
                    results.push((new_state.clone(), prob));
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
        
        let letter_count = self.factor_count();

        //If we have some results in the stash, return those first
        if let Some((new_state, new_prob)) = self.result_stash.pop() {
            self.state = new_state;
            self.current_val = new_prob;

            return self.state_to_result();
        }

        //Find the next configuration with the smallest incremental impact to probability
        if let Some(new_states) = self.find_smallest_next_increment() {
        
            //Advance the high-water mark for all returned states
            for (new_state, _new_prob) in new_states.iter() {
                for i in 0..letter_count {
                    if new_state[i] > self.high_water_mark[i] {
                        self.high_water_mark[i] = new_state[i];
                    }
                }
            }

            //Stash all the results we got
            self.result_stash = new_states;

            //Return one result from our stash
            let (new_state, new_prob) = self.result_stash.pop().unwrap();
            self.state = new_state;
            self.current_val = new_prob;

            return self.state_to_result();
                
        } else {
            //If we couldn't find any letter_slots to advancee, we've reached the end of the iteration
            return None;
        }
    }
}

// GOAT,
//3. Do a test with a function that does more than just multiplying all the factors together

//a. search for the word "letter", replace with "factor"
//b. search for the word "prob", replace with "element"
//√c. rename "permuted" to swizzled to avoid confusion

//Dust off that simplistic dictionary for words, to make a word_db lookup test & example

//Future needs:
//Sensitiveity map for radix iter.

//Make a LetterDistribution example 
