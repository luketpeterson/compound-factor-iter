
# compound-factor-iter Overview
This crate provides [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html) types that iterate all possible output permutations from a function that combines multiple discrete factors.

## What's it for?  For example?
Imagine you have a potential word, represented as a collection of letter probabilities.  Each letter of the word is a probability distribution that sums to `1.0`.  So the first letter might be `['a'=70%, 'b'=30%]`, and so on.  Now you want to find the most probable word, which is a permutation of the letter probabilities?

Sounds simple enough, just sort each individual letter's probability list and take the first one from each letter (aka factor).  How about the second-most-probable overall word?  Find the smallest difference in probability between the second most probable letter and the most probable letter, and substitute that letter.  How about the 1000th most probable?  Uhhhhhhh....?  That's what this crate is for.

## Usage
```rust
use compound_factor_iter::*;

fn idx_to_char(idx: usize) -> char {
    char::from_u32((idx+97) as u32).unwrap()
}

fn char_to_idx(c: char) -> usize {
    (c as usize) - 97
}

/// "bat", "cat", "hat", "bam", "cam", "ham"
let mut letter_probs = [[0.0; 26]; 3]; //3 letters, 26 possibilities each
letter_probs[0][char_to_idx('b')] = 0.4;
letter_probs[0][char_to_idx('c')] = 0.35;
letter_probs[0][char_to_idx('h')] = 0.25;
letter_probs[1][char_to_idx('a')] = 1.0;
letter_probs[2][char_to_idx('m')] = 0.35;
letter_probs[2][char_to_idx('t')] = 0.65;

let product_fn = |probs: &[f32]|{

    let mut new_prob = 1.0;
    for prob in probs.into_iter() {
        new_prob *= prob;
    }

    if new_prob > 0.0 {
        Some(new_prob)
    } else {
        None
    }
};

for (permutation, prob) in OrderedPermutationIter::new(letter_probs.iter(), &product_fn) {
    let word: String = permutation.into_iter().map(|idx| idx_to_char(idx)).collect();
    println!("permutation = {}, prob = {}", word, prob);
}
```

Using the [ManhattanPermutationIter] or the [RadixPermutationIter] is exactly the same.

```rust ignore
for (permutation, prob) in ManhattanPermutationIter::new(letter_probs.iter(), &product_fn) {
    let word: String = permutation.into_iter().map(|idx| idx_to_char(idx)).collect();
    println!("permutation = {}, prob = {}", word, prob);
}
```

## Monotonicity Requirement
The iterators in this crate can be used for a variety of functions that take discrete value as inputs, however one condition must hold: **An increase in any input factor's value must produce an increase in the combined function's output value.**  In other words, factors that can have a negative influence are not allowed.

## Why are there 3 different Iterators?
All 3 iterator types have exactly the same interface and can be used interchangeably, but they have vastly different performance and quality characteristics.

### OrderedPermutationIter
The [OrderedPermutationIter] is guaranteed to return results in order, from highest to lowest.  However, it can be **insanely** expensive because it needs to invoke the `combination_fn` closure potentially `n*2^n` times at each step, where `n` is the number of factors.

Due to the cost, the OrderedPermutationIter is only for situaitions where out-of-order results are unaccaptable.

### ManhattanPermutationIter
The [ManhattanPermutationIter] is a fixed-cost iterator that uses a simple heuristic to systematically explore outward from the known best permutation.  Unlike OrderedPermutationIter, ManhattanPermutationIter may return results out of order, however the heuristic ensures the results
mainly trend lower as the iteration progresses.

The word "Manhattan" comes from the fact that permutations are tried in an order determined by their [Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry) from the single best permutation.

**ManhattanPermutationIter is the best choice in most situations.**

### RadixPermutationIter
The [RadixPermutationIter] is another fixed-cost iterator implemented as a counter in a [Mixed Radix](https://en.wikipedia.org/wiki/Mixed_radix) number space.

In the majority of situations, the [ManhattanPermutationIter] will produce a higher quality sequence, but the [RadixPermutationIter] is appropriate when some factors have vastly more influence over the function result than others.  This can be caused by the function itself or by the input factor values, so it is recommended to try both iterators to determine the best one for your situation.

## The letter_distribution feature

Build with the `--features letter_distribution` feature enabled to get a handy-dandy object for representing a distribution of alphabetic letters, like what is described in the examples above.  This is used in many of the tests

## More Examples

Many more examples can be found by looking at the [tests.rs](https://github.com/luketpeterson/compound-factor-iter/blob/main/src/tests.rs) file.

Check out the `search_dict_test()` function for a good example using `letter_distribution`.

## Future Work

* Reversible option.  Currently all iterators iterate from highest value to lowest value.  In some situations it may be desirable to iterate from lowest value to highest.
