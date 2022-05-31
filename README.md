
# compound-factor-iter Overview
This crate provides [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html) types that iterate all possible output permutations from a function that combines multiple discrete factors.

## What's it for?  For example?
Imagine you have a potential word, represented as a collection of letter probabilities.  Each letter of the word is a probability distribution that sums to `1.0`.  So the first letter might be `['a'=70%, 'b'=30%]`, and so on.  Now you want to find the most probable word, which is a permutation of the letter probabilities?

Sounds simple enough, just sort each individual letter's probability list and take the first one from each letter (aka factor).  How about the second-most-probable overall word?  Find the smallest difference in probability between the second most probable letter and the most probable letter, and substitute that letter.  How about the 1000th most probable?  Uhhhhhhh....?  That's what this crate is for.

## Usage

GOAT

## Why are there 3 Different Iterators?

All 3 iterator types have exactly the same interface and can be used interchangeably, but they have vastly different performance and quality characteristics.

### OrderedPermutationIter
The [OrderedPermutationIter] is guaranteed to return results in order, from highest to lowest.  However, it can be **insanely** expensive because it needs to invoke the `combination_fn` closure potentially `n*2^n` times at each step, where `n` is the number of factors.

In general, the OrderedPermutationIter is only for situaitions where out-of-order results are unaccaptable.

GOAT, move this explanation to the Iterator's description page

### ApproxPermutationIter

GOAT

## More Examples

GOAT, SEE TESTS

## Future Work

