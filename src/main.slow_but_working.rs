
use core::fmt;
use std::fs;
use std::io::{BufReader, BufRead};
use std::path::{Path};
use std::cmp::Ordering;

use rand::prelude::*;
use rand_pcg::Pcg64;

/// How do you find the most probable overall outcome from a dozen individual probabilities?  Sounds
/// simple enough, just sort each individual probability and take the first one from each component.
/// How about the second-most-probable overall outcome?  Find the largest value among the second most
/// probable individual probabilities and substitute it in.  How about the 1000th most probable?
/// Uhhhhhhh....
/// 
/// That's what this crate is for.  [goat] iterates all compound probabilities composed from individual
/// component probabilities in the order of compound probability.
/// 

//GOAT TODO:
// 1.) Do tops-first
// 2.) finish migrating tests
// 3.) implement radix ordering
// 4.) Integrate fuzzy_rocks



fn main() {

    let _dict_tree = build_tree_from_dict("/usr/share/dict/words");
    

    //MAIN

    //Iterate over the possible words from the distribution,
    // compare them against the dict,
    // and stop when we get to 100 hits
    // for (i, (possible_word, word_prob)) in test_dist.permutations().enumerate() {
    //     println!("--{}: {:?} {}", i, possible_word, word_prob);
    // }

}

const BRANCHING_FACTOR: usize = 26;  //26 letters in the alphabet

#[derive(Debug, Clone, Default)]
pub struct LetterDistribution {
    /// The probability of each possible letter in each position.  Each inner array summs to 1.0
    letter_probs: Vec<[f32; BRANCHING_FACTOR]>,

    /// The indices into letter_probs, sorted by probability with the most probable in index 0
    sorted_letters: Vec<[usize; BRANCHING_FACTOR]>,
}

impl LetterDistribution {
    /// Makes a LetterDistribution from a set of specified probabilities
    pub fn from_probs(input_probs: &[Vec<(char, f32)>]) -> Self {

        let mut letter_probs = Vec::with_capacity(input_probs.len());
        for prob_column in input_probs.iter() {
            let mut new_letter = <[f32; BRANCHING_FACTOR]>::default();
            for (letter_char, letter_prob) in prob_column.iter() {
                new_letter[char_to_idx(*letter_char).unwrap()] = *letter_prob;
            }
            letter_probs.push(new_letter)
        }

        let mut new_dist = Self {
            letter_probs,
            sorted_letters: vec![],
        };
        new_dist.normalize_and_sort();
        new_dist
    }
    /// Makes a random LetterDistribution, used for testing.  A random distribution is
    /// probably the hardest possible case
    pub fn random<F: Fn(usize, usize, &mut Pcg64)->f32>(letter_count: usize, active_letters: usize, rng: &mut Pcg64, f: F) -> Self {

        let mut letter_probs = Vec::with_capacity(letter_count);
        for j in 0..letter_count {
            //Init with 0.0
            let mut new_letter: [f32; BRANCHING_FACTOR] = <_>::default();
            
            //Select up to active_letters at random to give non-zero prob to
            for i in 0..active_letters {
                let letter_idx: usize = rng.gen_range(0..BRANCHING_FACTOR);
                new_letter[letter_idx] = f(i, j, rng);
            }
            
            letter_probs.push(new_letter);
        }

        let mut new_dist = Self {
            letter_probs,
            sorted_letters: vec![],
        };
        new_dist.normalize_and_sort();
        new_dist
    }
    pub fn letter_count(&self) -> usize {
        self.letter_probs.len()
    }
    /// Returns an iterator that will generate the possible strings from a LetterDistribution,
    /// in descending order of probability, along with their probability
    pub fn ordered_permutations(&self) -> OrderedPermutationIter {
        OrderedPermutationIter::new(self)
    }
    fn normalize_and_sort(&mut self) {

        //Normalize so the probability for each letter summs to 1.0
        for distribution in self.letter_probs.iter_mut() {
            let mut sum = 0.0;
            for i in 0..BRANCHING_FACTOR {
                sum += distribution[i];
            }
            for i in 0..BRANCHING_FACTOR {
                distribution[i] /= sum;
            }
        }

        //Create a parallel array, sorted by the probability of each letter in descending order
        for letter in self.letter_probs.iter() {
            let mut sorted_letters = (0..BRANCHING_FACTOR).collect::<Vec<usize>>();
            sorted_letters.sort_by(|&letter_idx_a, &letter_idx_b| letter[letter_idx_b].partial_cmp(&letter[letter_idx_a]).unwrap_or(Ordering::Equal));
            self.sorted_letters.push(sorted_letters.try_into().unwrap());
        }

    }
}

pub struct OrderedPermutationIter<'a> {

    /// A reference to the distribution we're iterating over
    dist: &'a LetterDistribution,
    
    /// The current position of the result, as indices into the sorted_letters arrays
    state: Vec<usize>,

    /// The highest value state has achieved for a given letter
    high_water_mark: Vec<usize>,

    /// The threshold probability, corresponding to the last returned result
    current_prob: f32,

    /// A place to stash future results with probs that equal to the last-returned result
    result_stash: Vec<(Vec<usize>, f32)>,

    /// Initialization is a degenerate case
    new_iter: bool,
}

impl<'a> OrderedPermutationIter<'a> {
    fn new(dist: &'a LetterDistribution) -> Self {

        let letter_count = dist.letter_count();

        Self {
            dist,
            state: vec![0; letter_count],
            high_water_mark: vec![0; letter_count],
            current_prob: 1.0,
            result_stash: vec![],
            new_iter: true,
        }
    }
    fn state_to_result(&self) -> Option<(Vec<usize>, f32)> {
        
        let result = self.state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_letter_idx)| self.dist.sorted_letters[slot_idx][*sorted_letter_idx])
            .collect();

        //return None if prob is below ZERO_THRESHOLD
        const ZERO_THRESHOLD: f32 = 0.0000000001;
        if self.current_prob > ZERO_THRESHOLD {
            Some((result, self.current_prob))
        } else {
            None
        }
    }
    /// NOTE: we perform the arithmetic in 64-bit, even though we only care about a 32-bit
    /// result, because we need the value to be very, very stable, or we run the risk of
    /// ending up in an infinite loop or skipping a result
    fn prob_from_state(&self, state: &[usize]) -> f64 {
        let mut new_prob = 1.0;
        for (slot_idx, &sorted_letter_idx) in state.iter().enumerate() {

            let letter_idx = self.dist.sorted_letters[slot_idx][sorted_letter_idx];
            new_prob *= self.dist.letter_probs[slot_idx][letter_idx] as f64;
        }

        new_prob
    }
    /// Searches the frontier around a state, looking for the next state that has the highest overall
    /// probability, that is lower than the prob_threshold.  Returns None if it's impossible to advance
    /// to a non-zero probability.
    /// 
    fn find_smallest_next_increment(&self) -> Option<Vec<(Vec<usize>, f32)>> {

        let letter_count = self.dist.letter_count();

        let mut highest_prob = 0.0;
        let mut return_val = None;

        //The "tops" are the highest values each individual letter could possibly have and still reference
        // the next combination in the sequence
        let mut tops = Vec::with_capacity(letter_count);
        for &val in self.high_water_mark.iter() {
            tops.push((val+1).min(BRANCHING_FACTOR));
        }

        //GOAT ARGH!  There must be a way to establish "bottoms" that aren't so conservative as to be
        // paractially useless.  The commented out code below works, but in every test case, it only
        // finds bottoms that are equal to `vec![0; letter_count]`, which is pointless.
        //
        //We could recalculate bottom for each top permutation, except that the cost of calculating
        // bottom would outtweigh the savings

        //1. Each top must have a max net increment of 1
        //2. Working from a top that's known valid, I should be able to figure out the bottoms, to guide
        //  the rest of the iteration


        //A. choose a top, by iterating linearly.  Just that one value beyond the high-water mark, and the others
        //      left at the hwm.
        //B. choose all bottoms based on that top.
        //C. Iterate all permutations between the bottoms and the top,
        //      but (maybe) exclude testing permutations where the chosen top isn't at state or beyond...



        // //Find the "bottoms", i.e. the lowest value each letter could possibly have without exceeding
        // // prob_threshold
        // let mut bottoms = Vec::with_capacity(letter_count);
        // for i in 0..letter_count {
        //     let old_top = tops[i];
        //     let mut new_bottom = state[i];
        //     loop {
        //         if new_bottom == 0 {
        //             bottoms.push(0);
        //             break;
        //         }
        //         tops[i] = new_bottom; //Temporarily hijacking tops
        //         let prob = self.prob_from_state(&tops) as f32;
        //         if prob > prob_threshold {
        //             bottoms.push(new_bottom+1);
        //             break;
        //         } else {
        //             new_bottom -= 1;
        //         }
        //     }
        //     tops[i] = old_top;
        // }

        //We need to check every combination of adjustments between tops and bottoms
        let mut temp_state = vec![0; letter_count];//bottoms.clone();
        let mut finished = false;
        while !finished {

            //Increment the adjustments to the next state we want to try
            temp_state[0] += 1; //It is impossible for the initial starting case (all bottoms) to be the
                //next sequence element, because it's going to be the current sequence element or something
                //earlier
            let mut cur_letter = 0;
            while temp_state[cur_letter] > tops[cur_letter] {
                temp_state[cur_letter] = 0;//bottoms[cur_letter];
                cur_letter += 1;
                if cur_letter < letter_count {
                    temp_state[cur_letter] += 1;
                } else {
                    finished = true;
                    break;
                }
            }

            if !finished {

                let temp_prob = self.prob_from_state(&temp_state) as f32;
    
// println!("goatie {:?} is {}", temp_state, temp_prob);

                if temp_prob > 0.0 && temp_prob < self.current_prob && temp_prob >= highest_prob {

                    //Replace the results with a fresh array
                    if temp_prob > highest_prob {

                        //Advance "tops", so we'll find additional equal-weight results
                        //but first reset "tops" to the original value
                        for i in 0..letter_count {
                            tops[i] = (self.high_water_mark[i]+1).min(BRANCHING_FACTOR);
                            if temp_state[i] >= tops[i] {
                                tops[i] = temp_state[i]+1;
                            }
                        }

                        highest_prob = temp_prob;
                        return_val = Some(vec![(temp_state.clone(), highest_prob)]);

                    } else {
                        //We can infer temp_prob == highest_prob if we got here

                        //Advance "tops", so we'll find additional equal-weight results
                        for i in 0..letter_count {
                            if temp_state[i] >= tops[i] {
                                tops[i] = temp_state[i]+1;
                            }
                        }

                        //Append to the existing results array
                        return_val.as_mut().unwrap().push((temp_state.clone(), highest_prob));
                    }
                }
            }
        }

        return_val
    }
}

//GOAT.  Explanation to Adam.  I ended up going back to the original algorithm.  The reason I spent a few
// hours on that bitcode idea was because I was uneasy with the numerical instability that comes from doing
// a floating point compare on the results of arithmetic operations.  And I didn't feel like performing the
// numerical analysis to make it robust... However, I realized I could just perform the arithmetic in
// 64-bit, and then the comparison in 32 bit, and it should be stable for all intents and purposes.


//NOTE: This Iterator approach is possible because the letter probabilities are independent
// of each other, so the total probability is simply a multiplication of each one.
impl Iterator for OrderedPermutationIter<'_> {
    type Item = (Vec<usize>, f32);

    fn next(&mut self) -> Option<Self::Item> {
        
        let letter_count = self.dist.letter_probs.len();

        //If we have some results in the stash, return those first
        if let Some((new_state, new_prob)) = self.result_stash.pop() {
            self.state = new_state;
            self.current_prob = new_prob;

// println!("goat from_stash {:?} = {}", self.state, self.current_prob);
            return self.state_to_result();        
        }

        //Find the next configuration with the smallest incremental impact to probability
        if let Some(new_states) = self.find_smallest_next_increment() {
        
            //Advance the high-water mark for all returned states
            for (new_state, _new_prob) in new_states.iter() {
                for i in 0..letter_count {
                    if new_state[i] > self.high_water_mark[i] {
// println!("GOAT ADVANCE!!!! slot {}, {} to {}", i, self.high_water_mark[i], new_state[i]);
                        self.high_water_mark[i] = new_state[i];
                    }
                }
            }

            //Stash all the results we got
            self.result_stash = new_states;

            //Handle the startup case
            if self.new_iter {
                self.state = vec![0; letter_count]; //This should already be the case
                self.current_prob = self.prob_from_state(&vec![0; letter_count]) as f32;
                self.new_iter = false;
            } else {
                //Return one result from our stash
                let (new_state, new_prob) = self.result_stash.pop().unwrap();
                self.state = new_state;
                self.current_prob = new_prob;
            }

// println!("goat, new state {:?} = {}", self.state, self.current_prob);
            return self.state_to_result();
                
        } else {
            //If we couldn't find any letter_slots to advancee, we've reached the end of the iteration
            return None;
        }
    }
}

impl fmt::Display for LetterDistribution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "     -1-    -2-    -3-    -4-    -5-    -6-    -7-    -8-    -9-   -10-   -11-   -12-")?;
        for i in 0..BRANCHING_FACTOR {
            write!(f, "{} - ", char::from_u32((i+97) as u32).unwrap())?;
            for letter_prob in self.letter_probs.iter() {
                write!(f, "{:0.3}  ", letter_prob[i])?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct Tree {
    children: [Option<Box<Tree>>; BRANCHING_FACTOR],
}

//Returns None for spaces, punctuation, etc.
fn char_to_idx(c: char) -> Option<usize> {

    if c.is_ascii() {
        let lc_char = c.to_ascii_lowercase();
        let byte = lc_char as u8;
        if byte > 96 && byte < 123 {
            Some((byte - 97) as usize)
        } else {
            None
        }
    } else {
        None
    }
}

fn build_tree_from_dict<P: AsRef<Path>>(file_path: P) -> Tree {

    let f = fs::File::open(file_path).unwrap();

    let mut tree_root = Tree::default();

    //Read each line in the file, one word per line
    for line_result in BufReader::new(f).lines() {

        let mut cur_tree_node = &mut tree_root;

        //Iterate every char in the line
        for cur_char in line_result.unwrap().chars() {
            if let Some(char_idx) = char_to_idx(cur_char) {

                //A Some value for a children array-element means that this letter exists 
                if cur_tree_node.children[char_idx].is_none() {
                    cur_tree_node.children[char_idx] = Some(Box::new(Tree::default()));
                }
                cur_tree_node = cur_tree_node.children[char_idx].as_mut().unwrap();
            }
        }
    }

    tree_root
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::*;

    /// Convenience function for test cases
    fn group_result_by_prob(results: Vec<(Vec<usize>, f32)>) -> HashMap<String, Vec<Vec<usize>>> {

        let mut return_map = HashMap::new();

        for (result, prob) in results {
            let entry_list = return_map.entry(format!("{}", prob)).or_insert(vec![]);
            entry_list.push(result);
        }
        return_map
    }

    /// Convenience function for test cases
    #[cfg(test)]
    fn result_from_str(input: &str) -> Vec<usize> {
        let mut result = Vec::with_capacity(input.len());
        for c in input.chars() {
            result.push(char_to_idx(c).unwrap())
        }
        result
    }

    /// Convenience function for test cases
    #[cfg(test)]
    fn compare_grouped_results(a_group: HashMap<String, Vec<Vec<usize>>>, b_group: HashMap<String, Vec<Vec<usize>>>) -> bool {

        let mut sorted_a_group: Vec<(String, Vec<Vec<usize>>)> = a_group.into_iter().collect();
        sorted_a_group.sort_by(|(key_a, _group_a), (key_b, _group_b)| key_a.partial_cmp(key_b).unwrap_or(Ordering::Equal));

        let mut sorted_b_group: Vec<(String, Vec<Vec<usize>>)> = b_group.into_iter().collect();
        sorted_b_group.sort_by(|(key_a, _group_a), (key_b, _group_b)| key_a.partial_cmp(key_b).unwrap_or(Ordering::Equal));

        for ((_key_a, group_a), (_key_b, group_b)) in sorted_a_group.into_iter().zip(sorted_b_group.into_iter()) {
            
            //The groups may be in an arbitrary order, but if they have the same number of elements,
            // and every element from a is in b, we can be sure they're the same
            if group_a.len() != group_b.len() {
                return false;
            }

            for key_a in group_a {
                if !group_b.contains(&key_a) {
                    return false;
                }
            }
        }

        true
    }

    #[test]
    /// Test case to make sure we hit all 8 permutations in the right order.  This is the easiest test
    /// because all permutations have unique probabilities, and there are very few of them
    /// aaa=0.336, aab=0.224, aba=0.144, abb=0.096, baa=0.084, bab=0.056, bba=0.036, bbb=0.024
    fn test_0() {

        let letter_probs = vec![
            vec![('a', 0.8), ('b', 0.2)],
            vec![('a', 0.7), ('b', 0.3)],
            vec![('a', 0.6), ('b', 0.4)],
        ];
        let test_dist = LetterDistribution::from_probs(&letter_probs);
        println!("Testing:");
        println!("{}", test_dist);

        let results: Vec<(usize, (Vec<usize>, f32))> = test_dist.ordered_permutations().enumerate().collect();
        for (i, (possible_word, word_prob)) in results.iter() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        let result_strings: Vec<Vec<usize>> = results.into_iter().map(|(_idx, (string, _prob))| string).collect();
        assert_eq!(result_strings,
            vec![
                result_from_str("aaa"),
                result_from_str("aab"), 
                result_from_str("aba"),
                result_from_str("abb"),
                result_from_str("baa"),
                result_from_str("bab"),
                result_from_str("bba"),
                result_from_str("bbb"),
            ]);
    }

    #[test]
    /// Similar to test_0, but with some equal weights, to make sure tie-breaking logic works
    /// aaa=0.343, baa=0.147, aba=0.147, aab=0.147, abb=0.063, bab=0.063, bba=0.063, bbb=0.027
    fn test_1() {

        let letter_probs = vec![
            vec![('a', 0.7), ('b', 0.3)],
            vec![('a', 0.7), ('b', 0.3)],
            vec![('a', 0.7), ('b', 0.3)],
        ];
        let test_dist = LetterDistribution::from_probs(&letter_probs);
        println!("Testing:");
        println!("{}", test_dist);

        let results: Vec<(usize, (Vec<usize>, f32))> = test_dist.ordered_permutations().enumerate().collect();
        for (i, (possible_word, word_prob)) in results.iter() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        //Comparing floats is a pain... Just testing we get the right number of results for now
        assert_eq!(results.len(), 8);
    }

    #[test]
    /// "bat"=.233, "cat"=.233, "hat"=.233, "bam"=.100, "cam"=.100, "ham"=.100
    fn test_2() {

        let letter_probs = vec![
            vec![('b', 0.33), ('c', 0.33), ('h', 0.33)],
            vec![('a', 1.0)],
            vec![('m', 0.3), ('t', 0.7)],
        ];
        let test_dist = LetterDistribution::from_probs(&letter_probs);
        println!("Testing:");
        println!("{}", test_dist);

        let results: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().collect();
        for (i, (possible_word, word_prob)) in results.iter().enumerate() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        let grouped_results = group_result_by_prob(results);
        let grouped_truth = group_result_by_prob(
            vec![
                (result_from_str("bat"), 0.233),
                (result_from_str("cat"), 0.233),
                (result_from_str("hat"), 0.233),
                (result_from_str("bam"), 0.100),
                (result_from_str("cam"), 0.100),
                (result_from_str("ham"), 0.100),]
        );

        assert!(compare_grouped_results(grouped_results, grouped_truth));
    }

    #[test]
    /// Test case with multipe equal-weight possibilities, to test backtracking multiple positions
    ///  in a single step
    /// output should be: ac=0.16, xx=0.06 times 8, xx=0.0225 times 16
    fn test_3() {

        let letter_probs = vec![
            vec![('a', 0.4), ('b', 0.15), ('c', 0.15), ('d', 0.15), ('e', 0.15)],
            vec![('a', 0.15), ('b', 0.15), ('c', 0.4), ('d', 0.15), ('e', 0.15)],
        ];
        let test_dist = LetterDistribution::from_probs(&letter_probs);
        println!("Testing:");
        println!("{}", test_dist);

        //GOAT, temp debug print
        // for (i, (possible_word, word_prob)) in test_dist.permutations().enumerate() {
        //     println!("--{}: {:?} {}", i, possible_word, word_prob);
        // }

        let results: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().collect();
        for (i, (possible_word, word_prob)) in results.iter().enumerate() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        let grouped_results = group_result_by_prob(results);
        let grouped_truth = group_result_by_prob(
            vec![
                (result_from_str("ac"), 0.16),

                (result_from_str("aa"), 0.06),
                (result_from_str("ab"), 0.06),
                (result_from_str("ad"), 0.06),
                (result_from_str("ae"), 0.06),
                (result_from_str("bc"), 0.06),
                (result_from_str("cc"), 0.06),
                (result_from_str("dc"), 0.06),
                (result_from_str("ec"), 0.06),

                (result_from_str("ba"), 0.0225),
                (result_from_str("bb"), 0.0225),
                (result_from_str("bd"), 0.0225),
                (result_from_str("be"), 0.0225),
                (result_from_str("ca"), 0.0225),
                (result_from_str("cb"), 0.0225),
                (result_from_str("cd"), 0.0225),
                (result_from_str("ce"), 0.0225),
                (result_from_str("da"), 0.0225),
                (result_from_str("db"), 0.0225),
                (result_from_str("dd"), 0.0225),
                (result_from_str("de"), 0.0225),
                (result_from_str("ea"), 0.0225),
                (result_from_str("eb"), 0.0225),
                (result_from_str("ed"), 0.0225),
                (result_from_str("ee"), 0.0225),
                ]
        );

        assert!(compare_grouped_results(grouped_results, grouped_truth));
    }

    #[test]
    /// Test case with a nearly uniform distribution across several randomly chosen letters,
    /// with no exactly identical letter probabilities, and where probabilities correspond to
    /// their position in the probability matrix
    fn test_4() {

        let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
        let test_dist = LetterDistribution::random(4, 4, &mut rng, |i, j, _rng| 1.0 + (((i+1) as f32) / 10.0) + (((j+1) as f32) / 100.0));
        println!("{}", test_dist);
        
        //Print out the sorted probs, for debugging
        // for i in 0..test_dist.letter_count() {
        //     let sorted_probs: Vec<f32> = test_dist.sorted_letters[i].iter().take(4).map(|&idx| test_dist.letter_probs[i][idx]).collect();
        //     println!("*{}* {:?}", i, sorted_probs);
        // }

        //Compute each of the probs from a straightforward iteration, and then sort them
        // to produce a ground-truth results vector
        let mut ground_truth = vec![];
        let mut state = vec![0; 4];
        loop {

            let mut prob: f64 = 1.0;
            for l in 0..test_dist.letter_count() {
                let letter_idx = test_dist.sorted_letters[l][state[l]];
                prob *= test_dist.letter_probs[l][letter_idx] as f64;
            }

            let result: Vec<usize> = state.iter()
                .enumerate()
                .map(|(slot_idx, sorted_letter_idx)| test_dist.sorted_letters[slot_idx][*sorted_letter_idx])
                .collect();

            if prob > 0.0 {
                ground_truth.push((result, prob as f32));
            }
    
            if state == [3, 3, 3, 3] {
                break;
            }

            state[0] += 1;
            let mut cur_digit = 0;
            while state[cur_digit] > 3 {
                state[cur_digit] = 0;
                cur_digit += 1;
                state[cur_digit] += 1;
            }
        }
        ground_truth.sort_by(|(_, prob_a), (_, prob_b)| prob_b.partial_cmp(prob_a).unwrap_or(Ordering::Equal));
        // for (i, (state, prob)) in ground_truth.iter().enumerate() {
        //     println!("G--{} {:?} = {}", i, state, prob);
        // }

        //NOTE: some random seeds will have fewer results on account of chosen letter collisions,
        // but the seed we chose will produce all 256 results
        let results: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().collect();
        for (i, (possible_word, word_prob)) in results.iter().enumerate() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        assert_eq!(ground_truth, results);
    }

    #[test]
    /// The same idea as test_4, except with a ditribution of random rather than regular
    /// values.  This will hit cases where a single step on one letter might set another
    /// letter back many steps.  This violates the "conservation of net position" intuition.
    fn test_5() {

        let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
        let test_dist = LetterDistribution::random(4, 4, &mut rng, |_, _, rng| rng.gen());
        println!("{}", test_dist);
        
        //Compute each of the probs from a straightforward iteration, and then sort them
        // to produce a ground-truth results vector
        let mut ground_truth = vec![];
        let mut state = vec![0; 4];
        loop {

            let mut prob: f64 = 1.0;
            for l in 0..test_dist.letter_count() {
                let letter_idx = test_dist.sorted_letters[l][state[l]];
                prob *= test_dist.letter_probs[l][letter_idx] as f64;
            }

            let result: Vec<usize> = state.iter()
                .enumerate()
                .map(|(slot_idx, sorted_letter_idx)| test_dist.sorted_letters[slot_idx][*sorted_letter_idx])
                .collect();

            if prob > 0.0 {
                ground_truth.push((result, prob as f32));
            }
            
            if state == [3, 3, 3, 3] {
                break;
            }

            state[0] += 1;
            let mut cur_digit = 0;
            while state[cur_digit] > 3 {
                state[cur_digit] = 0;
                cur_digit += 1;
                state[cur_digit] += 1;
            }
        }
        ground_truth.sort_by(|(_, prob_a), (_, prob_b)| prob_b.partial_cmp(prob_a).unwrap_or(Ordering::Equal));
        // for (i, (state, prob)) in ground_truth.iter().enumerate() {
        //     println!("G--{} {:?} = {}", i, state, prob);
        // }

        let results: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().collect();
        for (i, (possible_word, word_prob)) in results.iter().enumerate() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
        }

        assert_eq!(ground_truth, results);
    }

    #[test]
    /// A random test distribution.  Random is a pathological case.
    fn test_6() {

        let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
        let test_dist = LetterDistribution::random(12, 4, &mut rng, |_, _, rng| rng.gen()); //GOAT, this is the real test
        //let test_dist = LetterDistribution::random(23, 4, &mut rng, |_, _, rng| rng.gen());
        println!("{}", test_dist);

        // Test that a subsequent result isn't more probable than a prior result
        let mut highest_prob = 1.0;
        let mut total_prob = 0.0;
        for (i, (possible_word, word_prob)) in test_dist.ordered_permutations().take(1000).enumerate() {
            println!("--{}: {:?} {}", i, possible_word, word_prob);
            if word_prob > highest_prob {
                println!("ERROR! i={}, {} > {}", i, word_prob, highest_prob);
                assert!(false);
            }
            total_prob += word_prob;
            highest_prob = word_prob;
        }
        println!("Total Distribution Prob Coverage: {}", total_prob);
    }

    //TODO: New test idea
    //Pick a random dictionary word, generate a prob distribution where every letter of that
    // word has ~50% probability with the other 50% being spread among 4 other random letters.
    //Then pick N letters in the distribution to swap the target letter with one of the minor
    // probabilities of a bogie letter chosen at random.

}