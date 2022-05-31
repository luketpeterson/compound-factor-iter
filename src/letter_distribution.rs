
use core::fmt;
use std::fs;
use std::io::{BufReader, BufRead};
use std::path::{Path, PathBuf};

use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::*;

//GOAT
// fn main() {

//     //Open the FuzzyRocks Table, or initialize it if it doesn't exist
//     let table = if !PathBuf::from("test.rocks").exists() {
//         let mut table = Table::<DefaultTableConfig, true>::new("test.rocks", DefaultTableConfig()).unwrap();
//         table.reset().unwrap();
//         init_table_with_dict(&mut table, "/usr/share/dict/words");
//         table
//     } else {
//         Table::<DefaultTableConfig, true>::new("test.rocks", DefaultTableConfig()).unwrap()
//     };

//     let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
//     //let test_dist = LetterDistribution::random(12, 4, &mut rng, |_, _, rng| rng.gen());
    
//     //"adventurous", on top of randomness
//     let mut test_dist = LetterDistribution::random(11, 3, &mut rng, |_, _, rng| rng.gen());
//     test_dist.set_letter_prob(0, 'a', 0.5);
//     test_dist.set_letter_prob(1, 'd', 0.5);
//     test_dist.set_letter_prob(2, 'v', 0.3);
//     test_dist.set_letter_prob(3, 'e', 0.5);
//     test_dist.set_letter_prob(4, 'n', 0.3);
//     test_dist.set_letter_prob(5, 't', 0.01);
//     test_dist.set_letter_prob(6, 'u', 0.5);
//     test_dist.set_letter_prob(7, 'r', 0.5);
//     test_dist.set_letter_prob(8, 'o', 0.01);
//     test_dist.set_letter_prob(9, 'u', 0.5);
//     test_dist.set_letter_prob(10, 's', 0.5);
//     println!("{}", test_dist);

//     //Iterate the permutations, and try looking each one up
//     //for (i, (permutation, prob)) in test_dist.radix_permutations().enumerate() {
//     for (i, (permutation, prob)) in test_dist.ordered_permutations().enumerate() {
        
//         let perm_string: String = permutation.into_iter().map(|idx| char::from((idx+97) as u8)).collect();
        
//         if i%100 == 0 {
//             println!("--{}: {:?} {}", i, perm_string, prob);
//         }
        
//         for (record_id, distance) in table.lookup_fuzzy(&perm_string, Some(2)).unwrap() {
            
//             //The value also happens to be the key.  How convenient.
//             let dict_word = table.get_value(record_id).unwrap();
//             println!("idx = {}, raw_prob = {}, {} matches {}, distance={}", i, prob, perm_string, dict_word, distance, );
//         }
//     }
// }

const BRANCHING_FACTOR: usize = 26;  //26 letters in the alphabet

#[derive(Debug, Clone, Default)]
pub struct LetterDistribution {
    /// The probability of each possible letter in each position.  Each inner array summs to 1.0
    letter_probs: Vec<[f32; BRANCHING_FACTOR]>,
}

impl LetterDistribution {
    pub fn set_letter_prob(&mut self, letter_idx: usize, letter: char, new_prob: f32) {

        let letter_ord = char_to_idx(letter).unwrap();
        let scale_factor = (1.0 - new_prob) / (1.0 - self.letter_probs[letter_idx][letter_ord]);

        //Scale the prob of all other possible letters in the distribution,
        // excluding the one we are about to set
        for (idx, prob) in self.letter_probs[letter_idx].iter_mut().enumerate() {
            if idx != letter_ord {
                *prob *= scale_factor;
            }
        }

        self.letter_probs[letter_idx][letter_ord] = new_prob;
        self.normalize();
    }
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
        };
        new_dist.normalize();
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
        };
        new_dist.normalize();
        new_dist
    }
    pub fn letter_count(&self) -> usize {
        self.letter_probs.len()
    }
    pub fn letter_probs(&self) -> &[[f32; BRANCHING_FACTOR]] {
        &self.letter_probs
    }
    /// Returns an iterator that will generate the possible strings from a LetterDistribution,
    /// in descending order of probability, along with their probability
    pub fn ordered_permutations(&self) -> OrderedPermutationIter<f32> {
        //PERF NOTE: Although the code in this closure is line-for-line IDENTICAL to the code in
        // Self::compound_probs, having this closure declared inline like this is about 15% faster.
        // My guess is that the compiler turns on some optimizations for inline closures that aren't
        // used when the function is declared elsewhere.
        OrderedPermutationIter::new(self.letter_probs.iter(), 1.0, &|probs|{

            // NOTE: we perform the arithmetic in 64-bit, even though we only care about a 32-bit
            // result, because we need the value to be very, very stable, or we run the risk of
            // ending up in an infinite loop or skipping a result
            let mut new_prob: f64 = 1.0;
            for prob in probs.iter() {
                new_prob *= *prob as f64;
            }
    
            if new_prob as f32 > 0.0 {
                Some(new_prob as f32)
            } else {
                None
            }
        })
    }
    pub fn manhattan_permutations(&self) -> ManhattanPermutationIter<f32> {
        ManhattanPermutationIter::new(self.letter_probs.iter(), &Self::compound_probs)
    }
    pub fn radix_permutations(&self) -> RadixPermutationIter<f32> {
        RadixPermutationIter::new(self.letter_probs.iter(), &Self::compound_probs)
    }
    fn normalize(&mut self) {

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
    }
    fn compound_probs(probs: &[f32]) -> Option<f32> {

        // NOTE: we perform the arithmetic in 64-bit, even though we only care about a 32-bit
        // result, because we need the value to be very, very stable, or we run the risk of
        // ending up in an infinite loop or skipping a result
        let mut new_prob: f64 = 1.0;
        for prob in probs.iter() {
            new_prob *= *prob as f64;
        }

        if new_prob as f32 > 0.0 {
            Some(new_prob as f32)
        } else {
            None
        }
    }
}

impl fmt::Display for LetterDistribution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "     -0-    -1-    -2-    -3-    -4-    -5-    -6-    -7-    -8-    -9-    -10-   -11-")?;
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

//Returns None for spaces, punctuation, etc.
pub fn char_to_idx(c: char) -> Option<usize> {

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

//GOAT
// fn init_table_with_dict<P: AsRef<Path>>(table: &mut Table::<DefaultTableConfig, true>, file_path: P) {

//     let f = fs::File::open(file_path).unwrap();

//     //Read each line in the file, one word per line
//     for (idx, line_result) in BufReader::new(f).lines().enumerate() {

//         let line = line_result.unwrap();
//         table.insert(line.clone(), &line).unwrap();

//         if idx % 1000 == 0 {
//             println!("Loaded word #{}, {}", idx, line);
//         }
//     }
// }
