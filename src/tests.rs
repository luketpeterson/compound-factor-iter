
use std::cmp::Ordering;
use std::collections::HashMap;

use rand::prelude::*;
use rand_pcg::Pcg64;

use crate::*;
use crate::letter_distribution::*;

//Compute each of the probs from a straightforward iteration, and then sort them
// to produce a ground-truth results vector
fn generate_ground_truth(dist: &LetterDistribution, set_letter_positions: usize) -> Vec<(Vec<usize>, f32)> {

    let sorted_letter_probs: Vec<Vec<(usize, f32)>> = dist.letter_probs().iter()
        .map(|factor_dist| {
            let mut sorted_elements: Vec<(usize, f32)> = factor_dist.as_ref().iter().cloned().enumerate().collect();
            sorted_elements.sort_by(|(_idx_a, element_a), (_idx_b, element_b)| element_b.partial_cmp(element_a).unwrap_or(Ordering::Equal));
            sorted_elements
        })
        .collect();

    let end_state = vec![set_letter_positions-1; dist.letter_count()];
    let mut ground_truth = vec![];
    let mut state = vec![0; dist.letter_count()];
    loop {

        let mut prob: f64 = 1.0;
        for l in 0..dist.letter_count() {
            let (letter_idx, _prob) = sorted_letter_probs[l][state[l]];
            prob *= dist.letter_probs()[l][letter_idx] as f64;
        }

        let result: Vec<usize> = state.iter()
            .enumerate()
            .map(|(slot_idx, sorted_letter_idx)| sorted_letter_probs[slot_idx][*sorted_letter_idx].0)
            .collect();

        if prob > 0.0 {
            ground_truth.push((result, prob as f32));
        }

        if state == end_state {
            break;
        }

        state[0] += 1;
        let mut cur_digit = 0;
        while state[cur_digit] > set_letter_positions-1 {
            state[cur_digit] = 0;
            cur_digit += 1;
            state[cur_digit] += 1;
        }
    }
    ground_truth.sort_by(|(_, prob_a), (_, prob_b)| prob_b.partial_cmp(prob_a).unwrap_or(Ordering::Equal));
    ground_truth
}

/// Convenience function for test cases
fn group_result_by_prob<T>(results: Vec<(Vec<usize>, T)>) -> HashMap<String, Vec<Vec<usize>>>
    where T: core::fmt::Display
{

    let mut return_map = HashMap::new();

    for (result, prob) in results {
        let entry_list = return_map.entry(format!("{}", prob)).or_insert(vec![]);
        entry_list.push(result);
    }
    return_map
}

/// Convenience function for test cases
fn result_from_str(input: &str) -> Vec<usize> {
    let mut result = Vec::with_capacity(input.len());
    for c in input.chars() {
        result.push(char_to_idx(c).unwrap())
    }
    result
}

/// Convenience function for test cases
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
fn ordered_test_0() {

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
fn ordered_test_1() {

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
fn ordered_test_2() {

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
fn ordered_test_3() {

    let letter_probs = vec![
        vec![('a', 0.4), ('b', 0.15), ('c', 0.15), ('d', 0.15), ('e', 0.15)],
        vec![('a', 0.15), ('b', 0.15), ('c', 0.4), ('d', 0.15), ('e', 0.15)],
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
fn ordered_test_4() {

    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    let test_dist = LetterDistribution::random(4, 4, &mut rng, |i, j, _rng| 1.0 + (((i+1) as f32) / 10.0) + (((j+1) as f32) / 100.0));
    println!("{}", test_dist);
    
    //Print out the sorted probs, for debugging
    // for i in 0..test_dist.letter_count() {
    //     let sorted_probs: Vec<f32> = test_dist.sorted_letters[i].iter().take(4).map(|&idx| test_dist.letter_probs[i][idx]).collect();
    //     println!("*{}* {:?}", i, sorted_probs);
    // }

    let ground_truth = generate_ground_truth(&test_dist, 4);
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
fn ordered_test_5() {

    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    let test_dist = LetterDistribution::random(4, 4, &mut rng, |_, _, rng| rng.gen());
    println!("{}", test_dist);
    
    let ground_truth = generate_ground_truth(&test_dist, 4);
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
fn ordered_test_6() {

    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    let test_dist = LetterDistribution::random(12, 4, &mut rng, |_, _, rng| rng.gen());
    // let test_dist = LetterDistribution::random(20, 4, &mut rng, |_, _, rng| rng.gen());
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

#[test]
/// A distribution with random values, but an exhaustible number of possible permutations,
/// in order to test boundary conditions
fn ordered_test_7() {
    let mut rng = Pcg64::seed_from_u64(1);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..4 {
        let mut inner_dist = vec![];
        for _ in 0..4 {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    println!("    -1-  -2-  -3-  -4-");
    for i in 0..4 {
        print!("{} -", i);
        for inner_dist in test_dist.iter() {
            print!("{:>4} ", inner_dist[i]);
        }
        println!("");
    }

    let perm_iter = OrderedPermutationIter::new(test_dist.iter(), &|products|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    });

    let mut highest_product = u32::MAX;
    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        if product > highest_product {
            println!("ERROR! i={}, {} > {}", i, product, highest_product);
            assert!(false);
        }
        highest_product = product;
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, 256);
}

#[test]
/// A distribution with an uneven number of elements in each factor
fn ordered_test_8() {
    let mut rng = Pcg64::seed_from_u64(3);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..3 {
        let dist_elements = rng.gen_range(1..8);
        let mut inner_dist = Vec::with_capacity(dist_elements);
        for _ in 0..dist_elements {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    let factor_element_counts: Vec<usize> = test_dist.iter().map(|inner| inner.len()).collect();
    let mut expected_perm_count = 1;
    factor_element_counts.iter().for_each(|cnt| expected_perm_count *= cnt);
    println!("\nfactor_element_counts {:?}", factor_element_counts);
    println!("expected_perm_count {}", expected_perm_count);

    let perm_iter = OrderedPermutationIter::new(test_dist.iter(), &|products|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    });

    let mut highest_product = u32::MAX;
    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        if product > highest_product {
            println!("ERROR! i={}, {} > {}", i, product, highest_product);
            assert!(false);
        }
        highest_product = product;
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, expected_perm_count);
}

#[test]
/// A basic test for the RadixPermutationIter
fn radix_test_0() {

    let letter_probs = vec![
        vec![('a', 0.8), ('b', 0.2)],
        vec![('a', 0.7), ('b', 0.3)],
        vec![('a', 0.6), ('b', 0.4)],
    ];
    let test_dist = LetterDistribution::from_probs(&letter_probs);
    println!("Testing:");
    println!("{}", test_dist);

    let results: Vec<(usize, (Vec<usize>, f32))> = test_dist.radix_permutations().enumerate().collect();
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
            result_from_str("bba"),
            result_from_str("bab"),
            result_from_str("bbb"),
        ]);
}

#[test]
/// Test the RadixPermutationIter with more than two possible options for each digit
fn radix_test_1() {

    let letter_probs = vec![
        vec![('a', 0.4), ('b', 0.3), ('c', 0.2), ('d', 0.1)],
        vec![('a', 0.4), ('b', 0.3), ('c', 0.2), ('d', 0.1)],
        vec![('a', 0.4), ('b', 0.3), ('c', 0.2), ('d', 0.1)],
    ];
    let test_dist = LetterDistribution::from_probs(&letter_probs);
    println!("Testing:");
    println!("{}", test_dist);
    
    let results: Vec<(Vec<usize>, f32)> = test_dist.radix_permutations().collect();
    for (i, (possible_word, word_prob)) in results.iter().enumerate() {
        println!("--{}: {:?} {}", i, possible_word, word_prob);
    }
    let grouped_results = group_result_by_prob(results);

    let ground_truth = generate_ground_truth(&test_dist, 4);
    // for (i, (possible_word, word_prob)) in ground_truth.iter().enumerate() {
    //     println!("G--{}: {:?} {}", i, possible_word, word_prob);
    // }
    let grouped_truth = group_result_by_prob(ground_truth);

    assert!(compare_grouped_results(grouped_results, grouped_truth));
}

#[test]
/// Compare a radix iterator against an ordered iterator. For this particular config should,
/// the top 100 radix results should contain all of the top 30 ordered results.  However,
/// this ratio is not constant / linear.  Roughly it seems it's more like a square relationship,
/// where the number of radix results needed to be certain you have all ordered results is
/// proportional to the square of the number of ordered results.
fn radix_test_2() {

    println!();
    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    let test_dist = LetterDistribution::random(12, 4, &mut rng, |_, _, rng| rng.gen());
    println!("{}", test_dist);

    let ordered: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().take(30).collect();
    let radix: Vec<(Vec<usize>, f32)> = test_dist.radix_permutations().take(100).collect();

    // let ordered: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().take(1000).collect();
    // let radix: Vec<(Vec<usize>, f32)> = test_dist.radix_permutations().take(3000).collect();

    let mut no_count = 0;
    for (i, (possible_word, word_prob)) in ordered.into_iter().enumerate() {
        if radix.contains(&(possible_word.clone(), word_prob)) {
            println!("YES --{}: {:?} {}", i, possible_word, word_prob);
        } else {
            println!("No --{}: {:?} {}", i, possible_word, word_prob);
            no_count += 1;
        }
    }

    assert_eq!(no_count, 0);
}

#[test]
/// A copy of ordered_test_7, except using the radix iterator
fn radix_test_3() {
    let mut rng = Pcg64::seed_from_u64(1);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..4 {
        let mut inner_dist = vec![];
        for _ in 0..4 {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    println!("    -1-  -2-  -3-  -4-");
    for i in 0..4 {
        print!("{} -", i);
        for inner_dist in test_dist.iter() {
            print!("{:>4} ", inner_dist[i]);
        }
        println!("");
    }

    let perm_iter = RadixPermutationIter::new(test_dist.iter(), &|products|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    });

    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, 256);
}

#[test]
/// A copy of ordered_test_8, except using the radix iterator
fn radix_test_4() {
    let mut rng = Pcg64::seed_from_u64(3);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..3 {
        let dist_elements = rng.gen_range(1..8);
        let mut inner_dist = Vec::with_capacity(dist_elements);
        for _ in 0..dist_elements {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    let factor_element_counts: Vec<usize> = test_dist.iter().map(|inner| inner.len()).collect();
    let mut expected_perm_count = 1;
    factor_element_counts.iter().for_each(|cnt| expected_perm_count *= cnt);
    println!("\nfactor_element_counts {:?}", factor_element_counts);
    println!("expected_perm_count {}", expected_perm_count);

    let product_fn = |products: &[u32]|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    };

    let perm_iter = RadixPermutationIter::new(test_dist.iter(), &product_fn);

    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, expected_perm_count);

    //Now compare the results against the ordered_permutations, which we'll use as the ground-truth
    let results: Vec<(Vec<usize>, u32)> = RadixPermutationIter::new(test_dist.iter(), &product_fn).collect();
    let grouped_results = group_result_by_prob(results);

    let ordered: Vec<(Vec<usize>, u32)> = OrderedPermutationIter::new(test_dist.iter(), &product_fn).collect();
    // for (i, (possible_word, word_prob)) in ordered.iter().enumerate() {
    //     println!("G--{}: {:?} {}", i, possible_word, word_prob);
    // }
    let grouped_truth = group_result_by_prob(ordered);

    assert!(compare_grouped_results(grouped_results, grouped_truth));
}

#[test]
/// A basic test for the ManhattanPermutationIter
fn manhattan_test_0() {

    let letter_probs = vec![
        vec![('a', 0.8), ('b', 0.2)],
        vec![('a', 0.7), ('b', 0.3)],
        vec![('a', 0.6), ('b', 0.4)],
    ];
    let test_dist = LetterDistribution::from_probs(&letter_probs);
    println!("Testing:");
    println!("{}", test_dist);

    let results: Vec<(usize, (Vec<usize>, f32))> = test_dist.manhattan_permutations().enumerate().collect();
    for (i, (possible_word, word_prob)) in results.iter() {
        println!("--{}: {:?} {}", i, possible_word, word_prob);
    }

    let result_strings: Vec<Vec<usize>> = results.into_iter().map(|(_idx, (string, _prob))| string).collect();
    assert_eq!(result_strings,
        vec![
            result_from_str("aaa"),
            result_from_str("aab"), 
            result_from_str("aba"),
            result_from_str("baa"), //NOTE: These are out of order, but we're testing Manhattan behavior
            result_from_str("abb"),
            result_from_str("bab"),
            result_from_str("bba"),
            result_from_str("bbb"),
        ]);
}

#[test]
/// Another basic test for the ApproxPermutationIter, but with more than 2 permutations per letter
fn manhattan_test_1() {

    let letter_probs = vec![
        vec![('a', 0.7), ('b', 0.2), ('c', 0.1)],
        vec![('a', 0.6), ('b', 0.3), ('c', 0.1)],
        vec![('a', 0.5), ('b', 0.4), ('c', 0.1)],
    ];
    let test_dist = LetterDistribution::from_probs(&letter_probs);
    println!("Testing:");
    println!("{}", test_dist);

    let results: Vec<(Vec<usize>, f32)> = test_dist.manhattan_permutations().collect();
    for (i, (possible_word, word_prob)) in results.iter().enumerate() {
        println!("--{}: {:?} {}", i, possible_word, word_prob);
    }
    let grouped_results = group_result_by_prob(results);

    let ground_truth = generate_ground_truth(&test_dist, 4);
    // for (i, (possible_word, word_prob)) in ground_truth.iter().enumerate() {
    //     println!("G--{}: {:?} {}", i, possible_word, word_prob);
    // }
    let grouped_truth = group_result_by_prob(ground_truth);

    assert!(compare_grouped_results(grouped_results, grouped_truth));
}

#[test]
/// Compare a manhattan iterator against an ordered iterator. For this particular config should,
/// the top 500 manhattan results should contain all of the top 50 ordered results.  However,
/// this ratio is not constant / linear.  Roughly it seems it's like the relationship is k*n log(n),
/// where n is the number of manhattan results needed to be very likely you have all ordered results.
fn manhattan_test_2() {

    println!();
    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    let test_dist = LetterDistribution::random(12, 4, &mut rng, |_, _, rng| rng.gen());
    println!("{}", test_dist);

    let ordered: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().take(350).collect();
    let radix: Vec<(Vec<usize>, f32)> = test_dist.manhattan_permutations().take(14000).collect();

    // let ordered: Vec<(Vec<usize>, f32)> = test_dist.ordered_permutations().take(1750).collect();
    // let radix: Vec<(Vec<usize>, f32)> = test_dist.manhattan_permutations().take(200000).collect();

    let mut no_count = 0;
    for (i, (possible_word, word_prob)) in ordered.into_iter().enumerate() {
        if radix.contains(&(possible_word.clone(), word_prob)) {
            println!("YES --{}: {:?} {}", i, possible_word, word_prob);
        } else {
            println!("No --{}: {:?} {}", i, possible_word, word_prob);
            no_count += 1;
        }
    }

    assert_eq!(no_count, 0);
}

#[test]
/// A copy of ordered_test_7, except using the manhattan iterator
fn manhattan_test_3() {
    let mut rng = Pcg64::seed_from_u64(1);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..4 {
        let mut inner_dist = vec![];
        for _ in 0..4 {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    println!("    -1-  -2-  -3-  -4-");
    for i in 0..4 {
        print!("{} -", i);
        for inner_dist in test_dist.iter() {
            print!("{:>4} ", inner_dist[i]);
        }
        println!("");
    }

    let perm_iter = ManhattanPermutationIter::new(test_dist.iter(), &|products|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    });

    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, 256);
}

#[test]
/// A copy of ordered_test_8, except using the manhattan iterator
fn manhattan_test_4() {
    let mut rng = Pcg64::seed_from_u64(3);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..3 {
        let dist_elements = rng.gen_range(1..8);
        let mut inner_dist = Vec::with_capacity(dist_elements);
        for _ in 0..dist_elements {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    let factor_element_counts: Vec<usize> = test_dist.iter().map(|inner| inner.len()).collect();
    let mut expected_perm_count = 1;
    factor_element_counts.iter().for_each(|cnt| expected_perm_count *= cnt);
    println!("\nfactor_element_counts {:?}", factor_element_counts);
    println!("expected_perm_count {}", expected_perm_count);

    let product_fn = |products: &[u32]|{

        let mut new_product: u32 = 1;
        for product in products.iter() {
            new_product *= *product;
        }

        Some(new_product)
    };

    let perm_iter = ManhattanPermutationIter::new(test_dist.iter(), &product_fn);

    let mut perm_cnt = 0;
    for (i, (perm, product)) in perm_iter.enumerate() {
        println!("--{}: {:?} {}", i, perm, product);
        perm_cnt += 1;
    }
    assert_eq!(perm_cnt, expected_perm_count);

    //Now compare the results against the ordered_permutations, which we'll use as the ground-truth
    let results: Vec<(Vec<usize>, u32)> = ManhattanPermutationIter::new(test_dist.iter(), &product_fn).collect();
    let grouped_results = group_result_by_prob(results);

    let ordered: Vec<(Vec<usize>, u32)> = OrderedPermutationIter::new(test_dist.iter(), &product_fn).collect();
    // for (i, (possible_word, word_prob)) in ordered.iter().enumerate() {
    //     println!("G--{}: {:?} {}", i, possible_word, word_prob);
    // }
    let grouped_truth = group_result_by_prob(ordered);

    assert!(compare_grouped_results(grouped_results, grouped_truth));
}

#[test]
/// Search a dictionary for a specific word
/// 
/// Make a noisy distribution around a dictionary word, where every letter of that word
/// has ~50% probability with the other 50% being spread among 2-4 other random letters.
/// Then pick N letters in the distribution to really mess up the leters so a bogie letter
/// has a much higher probability than the "correct" letter.
/// 
fn search_dict_test() {

    //Open the dictionary file
    let dict_tree = LetterTree::new_from_dict_file("/usr/share/dict/words");

    let mut rng = Pcg64::seed_from_u64(1); //non-cryptographic random used for repeatability
    
    //"adventurous", on top of randomness
    let mut test_dist = LetterDistribution::random(11, 3, &mut rng, |_, _, rng| rng.gen());
    test_dist.set_letter_prob(0, 'a', 0.5);
    test_dist.set_letter_prob(1, 'd', 0.5);
    test_dist.set_letter_prob(2, 'v', 0.3);
    test_dist.set_letter_prob(3, 'e', 0.5);
    test_dist.set_letter_prob(4, 'n', 0.3);
    test_dist.set_letter_prob(5, 't', 0.01);
    test_dist.set_letter_prob(6, 'u', 0.5);
    test_dist.set_letter_prob(7, 'r', 0.5);
    test_dist.set_letter_prob(8, 'o', 0.01);
    test_dist.set_letter_prob(9, 'u', 0.5);
    test_dist.set_letter_prob(10, 's', 0.5);
    println!("{}", test_dist);

    //NOTE: this test really shows the difference between the Manhattan and the Radix iterator.  The
    // Manhattan iter finds it in about 32K permutations, but the Radix iter takes 1.8M

    //Iterate the permutations, and try looking each one up
    for (i, (permutation, prob)) in test_dist.manhattan_permutations().enumerate().take(100000) {
    //for (i, (permutation, prob)) in test_dist.radix_permutations().enumerate().take(2000000) {
    //for (i, (permutation, prob)) in test_dist.ordered_permutations().enumerate() {
        
        let perm_string: String = permutation.into_iter().map(|idx| char::from((idx+97) as u8)).collect();
        
        // if i%100 == 0 {
        //     println!("--{}: {:?} {}", i, perm_string, prob);
        // }
        
        let matched_letters = dict_tree.search(&perm_string);
        
        if matched_letters+3 > perm_string.len() {
            println!("idx = {}, raw_prob = {}, {} matches {} letters", i, prob, perm_string, matched_letters);
        }
    }
}

#[test]
/// Tests iterators with non-multiplicative combination_fn
fn non_multiplicative_fn_test() {

    //A random 4^4 set (256 permutations)
    let mut rng = Pcg64::seed_from_u64(1);
    let mut test_dist: Vec<Vec<u32>> = vec![];
    for _ in 0..4 {
        let mut inner_dist = vec![];
        for _ in 0..4 {
            inner_dist.push(rng.gen_range(0..256));
        }
        test_dist.push(inner_dist);
    }

    println!("    -1-  -2-  -3-  -4-");
    for i in 0..4 {
        print!("{} -", i);
        for inner_dist in test_dist.iter() {
            print!("{:>4} ", inner_dist[i]);
        }
        println!("");
    }

    let non_multiplicative_fn = |values: &[u32]| {

        let mut new_value: u32 = 0;
        for (i, val) in values.iter().enumerate() {
            new_value += (i as u32) * 256 * val;
        }

        Some(new_value)
    };

    let ordered: Vec<(Vec<usize>, u32)> = OrderedPermutationIter::new(test_dist.iter(), &non_multiplicative_fn).collect();

    let mut ground_truth = ordered.clone();
    ground_truth.sort_by(|(_element_a, val_a), (_element_b, val_b)| val_b.partial_cmp(val_a).unwrap_or(Ordering::Equal));

    //Test the ordered permutations are identical to the ground truth
    assert_eq!(ordered, ground_truth);

    //Compare the Manhattan output quality using a distance-squared-error formula
    //Each out-of-place result adds the distance it is from its proper place in the sequence squared
    // to the compound error.
    //
    let manhattan: Vec<(Vec<usize>, u32)> = ManhattanPermutationIter::new(test_dist.iter(), &non_multiplicative_fn).collect();
    let mut total_err: u64 = 0;
    for (i, manhattan_result) in manhattan.iter().enumerate() {
        //println!("Manhattan --{}: {:?} {}", i, manhattan_result.0, manhattan_result.1);
        let truth_pos = ground_truth.iter().position(|element| element==manhattan_result).unwrap();
        total_err += (truth_pos.abs_diff(i) as u64).pow(2);
    }
    println!("Manhattan Total Squared-Error: {}", total_err);

    //And now do that for Radix
    let radix: Vec<(Vec<usize>, u32)> = RadixPermutationIter::new(test_dist.iter(), &non_multiplicative_fn).collect();
    let mut total_err: u64 = 0;
    for (i, radix_result) in radix.iter().enumerate() {
        //println!("Radix --{}: {:?} {}", i, manhattan_result.0, manhattan_result.1);
        let truth_pos = ground_truth.iter().position(|element| element==radix_result).unwrap();
        total_err += (truth_pos.abs_diff(i) as u64).pow(2);
    }
    println!("Radix Total Squared-Error: {}", total_err);
}