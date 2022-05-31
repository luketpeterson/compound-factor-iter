
use std::cmp::Ordering;

/// Builds orderings tables for a given distribution
/// 
/// A set of sorted lists of precedence for each factor, each representing an
///  ordering regime.  The idea is that the best ordering regime changes as the
///  iteration progresses.
/// 
/// - Ordering Regime 0
/// In the beginning, the factors that have the least influence on the combined value
/// (move the combined value the least from the maximum), are the factors with the smallest
/// difference between their maximum value and their second-highest value.
/// 
/// - Ordering Regime 1
/// Once all permutations of the highest and second-highest values for all factors have
/// been considered, Ordering Regime 1 is based on the absolute value of the second-
/// highest element.  This is because factors in this regime may take on values from their
/// highest, second-highest, and third-highest elements, so we want to postpone the small
/// third-highest elements to later in the sequence.
/// 
/// - Ordering Regime 2
/// The final ordering that prevails for the rest of the iteration is based on the sum of
/// the top-3 elements for every factor.
/// 

pub fn build_orderings<T>(sorted_dists: &Vec<Vec<(usize, T)>>, _combination_fn: &dyn Fn(&[T]) -> Option<T>) -> Vec<Vec<usize>>
    where
    T: Copy + PartialOrd + num_traits::Bounded + num_traits::Zero + core::ops::Sub<Output=T>,
{
    let factor_count = sorted_dists.len();

    let mut orderings: Vec<Vec<usize>> = Vec::with_capacity(3);
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

    //GOAT, temporarily here so we don't change regimes
    orderings.push(orderings[0].clone());
    orderings.push(orderings[0].clone());

    //GOAT, Real implementation of regimes 1 & 2
    // //After we pass [1, 1, 1, 1, 1, ...], but
    // //Before we make it to [2, 2, 2, 2, 2, ...]
    // orderings.push({
    //     //NOTE: For moderate distributions, the best results come from considering
    //     // the second-place value
    //     let mut ordering = Vec::with_capacity(factor_count);
    //     for i in 0..factor_count {        
    //         let prob = sorted_dists[i][1].1;

    //         ordering.push((prob, i));
    //     }
    //     ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
    //     let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
    //     ordering
    // });

    // //After we pass [2, 2, 2, 2, 2, ...]
    // orderings.push({
    //     //NOTE: For distributions with a small number of factors, it really shouldn't
    //     // matter much because we can easily iterate the whole set, but we get better
    //     // results considering the second, third, etc. (up to 3 in this case)
    //     let mut ordering = Vec::with_capacity(factor_count);
    //     for i in 0..factor_count {        

    //         let mut l = 1;
    //         let mut prob = T::zero();
    //         while l < factor_count && l < 3 && l < sorted_dists[i].len() {
    //             prob = prob + sorted_dists[i][l].1;
    //             l += 1;
    //         }

    //         ordering.push((prob, i));
    //     }
    //     ordering.sort_by(|(prob_a, _idx_a), (prob_b, _idx_b)| prob_b.partial_cmp(&prob_a).unwrap_or(Ordering::Equal));
    //     let ordering = ordering.into_iter().map(|(_prob, idx)| idx).collect();
    //     ordering
    // });

    orderings
}