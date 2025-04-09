(= (fact $n) 
    (if (<= $n 1)
        1
        (* $n (fact (- $n 1)))
    )
) 

(= (permutation $list $n) 
    (let $max-permutations (fact (size-atom $list))  ;; Calculate the factorial of the list size
        (if (> $max-permutations 1000)   ;; Arbitrary threshold to avoid combinatorial explosion
            ('permutation too big')  ;; Return an error if too many permutations
            (if (or (== $n 0) (== $list ()))
                ()  ;; Base case: no permutations
                (let* (
                    ($x (superpose $list))  ;; Non-deterministically pick one atom
                    ($rest (remove-one $list $x))  ;; Remove just one occurrence
                    ($next (permutation $rest (- $n 1)))  ;; Build rest of permutation
                )
                (cons-atom $x $next)  ;; Combine the current atom with the rest
                )
            )
        )
    )
) 

(= (remove-one $list $val)
    (if (== $list ())
        ()
        (let* (
            (($h $t) (decons-atom $list))
            ($rest (remove-one $t $val))
        )
        (if (== $h $val)
            $t  ;; Remove the first match
            (cons-atom $h $rest)  ;; Otherwise keep and continue
        )
        )
    )
)

!(permutation (1 3 4  4) 3)   

