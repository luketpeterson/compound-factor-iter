
use crate::*;

/// A fast permutation iterator based on traversal through a mixed-radix space.
/// This is much much faster than the ordered search, although it may return
/// some results out of order.
/// 

pub struct RadixPermutationIter<'a> {

    /// A reference to the distribution we're iterating over
    dist: &'a LetterDistribution,

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
    ordering: Vec<usize>,

    /// The current position of the result, as indices into the sorted_letters arrays
    state: Vec<usize>,

    /// The maximum digit position across the entire state vec.  This allows us to evenly
    /// increment the low positions of every digit before attempting to set any digits to
    /// relatively high (improbable) values.
    max_digit: usize,

    /// Initialization is a degenerate case
    new_iter: bool,
}

impl<'a> RadixPermutationIter<'a> {
    pub fn new(dist: &'a LetterDistribution) -> Self {

        let letter_count = dist.letter_count();

        let ordering = match letter_count {
            0..=7 => {
                //NOTE: For distributions with a small number of factors, it really shouldn't
                // matter much because we can easily iterate the whole set, but we get better
                // results considering the second, third, etc. (up to 3 in this case)
                let mut ordering = Vec::with_capacity(letter_count);
                for i in 0..letter_count {        

                    let mut l = 1;
                    let mut prob = 0.0;
                    while l < letter_count && l < 3 {
                        let idx = dist.sorted_letters[i][l];
                        prob += dist.letter_probs[i][idx];
                        l += 1;  
                    }
        
                    ordering.push((prob, i));
                }
                ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
                let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
                ordering
            },
            8..=16 => {
                //NOTE: For moderate distributions, the best results come from considering
                // the second-place value
                let mut ordering = Vec::with_capacity(letter_count);
                for i in 0..letter_count {        
                    let idx = dist.sorted_letters[i][1];
                    let prob = dist.letter_probs[i][idx];
        
                    ordering.push((prob, i));
                }
                ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
                let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
                ordering
            },
            _ => {
                //NOTE: It seems the best results on the nastiest distributions, i.e. with
                //  the most factors, come from establishing ordering based on the
                //  difference between the top and second places.
                let mut ordering = Vec::with_capacity(letter_count);
                for i in 0..letter_count {                    
                    let idx_0 = dist.sorted_letters[i][0];
                    let prob_0 = dist.letter_probs[i][idx_0];
        
                    let idx_1 = dist.sorted_letters[i][1];
                    let prob_1 = dist.letter_probs[i][idx_1];
        
                    ordering.push((prob_0 - prob_1, i));
                }
                ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_a.partial_cmp(&prob_b).unwrap_or(Ordering::Equal));
                let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
                ordering
            }
        };

        Self {
            dist,
            ordering,
            state: vec![0; letter_count],
            max_digit: 1,
            new_iter: true,
        }
    }
    //TODO: unify this with OrderedPermutationIter
    fn state_to_result(&self) -> Option<(Vec<usize>, f32)> {

        let mut permuted_state = vec![0; self.dist.letter_count()];
        for (i, &idx) in self.ordering.iter().enumerate() {
            permuted_state[i] = self.state[idx];
        }

        let prob = Self::prob_from_state(self.dist, &permuted_state) as f32;

        let result = permuted_state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_letter_idx)| {
                self.dist.sorted_letters[slot_idx][*sorted_letter_idx]
            })
            .collect();

        //return None if prob is below ZERO_THRESHOLD
        const ZERO_THRESHOLD: f32 = 0.0000000001;
        if prob > ZERO_THRESHOLD {
            Some((result, prob))
        } else {
            None
        }
    }
    //TODO: unify this with OrderedPermutationIter
    /// NOTE: we perform the arithmetic in 64-bit, even though we only care about a 32-bit
    /// result, because we need the value to be very, very stable, or we run the risk of
    /// ending up in an infinite loop or skipping a result
    ///
    fn prob_from_state(dist: &LetterDistribution, state: &[usize]) -> f64 {
        let mut new_prob = 1.0;
        for (slot_idx, &sorted_letter_idx) in state.iter().enumerate() {

            let letter_idx = dist.sorted_letters[slot_idx][sorted_letter_idx];
            new_prob *= dist.letter_probs[slot_idx][letter_idx] as f64;
        }

        new_prob
    }

    fn step(&mut self) -> (bool, Option<(Vec<usize>, f32)>) {

        let letter_count = self.dist.letter_count();

        //TODO, if a component reaches the zero threshold then that is the effective max_digit
        // for that component
        //TODO, if every component is at the zero threshold then we're done iterating
        
        self.state[0] += 1;
        let mut cur_digit = 0;
        while self.state[cur_digit] > self.max_digit {

            self.state[cur_digit] = 0;
            cur_digit += 1;

            if cur_digit < letter_count {
                self.state[cur_digit] += 1;
            } else {
                if self.max_digit < BRANCHING_FACTOR-1 {
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

impl Iterator for RadixPermutationIter<'_> {
    type Item = (Vec<usize>, f32);

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