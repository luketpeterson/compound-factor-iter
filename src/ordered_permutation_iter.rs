
use std::cmp::Ordering;

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
    current_prob: f32,

    /// A place to stash future results with probs that equal to the last-returned result
    result_stash: Vec<(Vec<usize>, f32)>,
}

//GOAT, T instead of f32
// impl<'a, T> OrderedPermutationIter<'a, T>
//     where
//     T: Clone + PartialOrd, //GOAT, it's likely I want U as the type that's part of the struct def, and not T
impl<'a> OrderedPermutationIter<'a, f32>
{
    //GOAT, T instead of f32
    //pub fn new<E: AsRef<[T]>, F: Fn(&[T]) -> Option<T>>(factor_iter: impl Iterator<Item=E>, combination_fn: &'a F) -> Self {
    pub fn new<E: AsRef<[f32]>, F: Fn(&[f32]) -> Option<f32>>(factor_iter: impl Iterator<Item=E>, combination_fn: &'a F) -> Self {

        //GOAT, T instead of f32
        //let mut sorted_dists: Vec<Vec<(usize, T)>> = factor_iter
        let sorted_dists: Vec<Vec<(usize, f32)>> = factor_iter
            .map(|factor_dist| {
                //GOAT, T instead of f32
                let mut sorted_elements: Vec<(usize, f32)> = factor_dist.as_ref().iter().cloned().enumerate().collect();
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
            current_prob: 1.0,
            result_stash: vec![],
        }
    }
    pub fn factor_count(&self) -> usize {
        self.sorted_dists.len()
    }
    //NOTE: A future optimization is to keep the factors vector in memory rather than
    // rebuilding it every time.
    //GOAT, T instead of f32
    fn sorted_state_combined_val(&self, state: &[usize]) -> Option<f32> {

        //GOAT, T instead of f32
        let factors: Vec<f32> = state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_idx)| self.sorted_dists[slot_idx][*sorted_idx].1)
            .collect();

        (self.combination_fn)(&factors)
    }
    //GOAT, T instead of f32
    fn state_to_result(&self) -> Option<(Vec<usize>, f32)> {
        
        let result: Vec<usize> = self.state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_letter_idx)| self.sorted_dists[slot_idx][*sorted_letter_idx].0)
            .collect();

        self.sorted_state_combined_val(&self.state)
            .map(|combined_val| (result, combined_val))
    }
    /// Searches the frontier around a state, looking for the next state that has the highest overall
    /// probability, that is lower than the prob_threshold.  Returns None if it's impossible to advance
    /// to a non-zero probability.
    /// 
    fn find_smallest_next_increment(&self) -> Option<Vec<(Vec<usize>, f32)>> {

        let factor_count = self.factor_count();

        let mut highest_prob = 0.0;
        let mut return_val = None;

        //GOAT TODO: Write up a better explanation of the alforithm overall, once it's fully debugged

        //NOTE: when letter_to_advance == factor_count, that means we don't attempt to advance any letter
        for letter_to_advance in 0..(factor_count+1) {

            //The "tops" are the highest values each individual letter could possibly have and still reference
            // the next combination in the sequence
            let mut tops = Vec::with_capacity(factor_count);
            for (i , &val) in self.high_water_mark.iter().enumerate() {
                if i == letter_to_advance {
                    tops.push((val+1).min(self.sorted_dists[i].len()));
                } else {
                    tops.push((val).min(self.sorted_dists[i].len()));
                }
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
                    let prob = self.sorted_state_combined_val(&tops);
                    if prob.is_some() && prob.unwrap() > self.current_prob {
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
            if letter_to_advance < factor_count {
                temp_state[letter_to_advance] = tops[letter_to_advance];
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
                    cur_letter = 0;
                } else {
                    temp_state[1] += 1;
                    cur_letter = 1;
                }

                //Deal with any rollover caused by the increment above
                while temp_state[cur_letter] > tops[cur_letter] {

                    temp_state[cur_letter] = bottoms[cur_letter];
                    cur_letter += 1;

                    //Skip over the letter_to_advance, which we're going to leave pegged to tops
                    if cur_letter == letter_to_advance {
                        cur_letter += 1;
                    }

                    if cur_letter < factor_count {
                        temp_state[cur_letter] += 1;
                    } else {
                        finished = true;
                        break;
                    }
                }
    
                if let Some(temp_prob) = self.sorted_state_combined_val(&temp_state) {
                    if temp_prob > 0.0 && temp_prob < self.current_prob && temp_prob >= highest_prob {

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
    fn find_adjacent_equal_permutations(&self, state: &[usize], prob: f32, results: &mut Vec<(Vec<usize>, f32)>) {

        let letter_count = self.factor_count();
        let mut new_state = state.to_owned();
        
        loop {

            new_state[0] += 1;
            let mut cur_digit = 0;
            let mut temp_prob = self.sorted_state_combined_val(&new_state);

            while temp_prob.is_some() && temp_prob.unwrap() < prob {

                new_state[cur_digit] = state[cur_digit];
                cur_digit += 1;

                if cur_digit == letter_count {
                    break;
                }

                new_state[cur_digit] += 1;
                temp_prob = self.sorted_state_combined_val(&new_state);
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

//GOAT, T instead of f32
impl Iterator for OrderedPermutationIter<'_, f32> {
    type Item = (Vec<usize>, f32);

    fn next(&mut self) -> Option<Self::Item> {
        
        let letter_count = self.factor_count();

        //If we have some results in the stash, return those first
        if let Some((new_state, new_prob)) = self.result_stash.pop() {
            self.state = new_state;
            self.current_prob = new_prob;

// println!("from_stash {:?} = {}", self.state, self.current_prob); //GOAT, debug print
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
            self.current_prob = new_prob;

            return self.state_to_result();
                
        } else {
            //If we couldn't find any letter_slots to advancee, we've reached the end of the iteration
            return None;
        }
    }
}



// GOAT,
//4. Do a test for different radixes, i.e. each factor containing different numbers of elements


//a. search for the word "letter", replace with "factor"
//b. search for the word "prob", replace with "element"
