(= (filter $f $list)
  (if (== $list ())
      ()  ;; Base case: return an empty list
      (let* (
        (($h $t) (decons-atom $list))  ;; Deconstruct the list into head and tail
        ($rest (filter $f $t))  ;; Recursively filter the tail
      )
        (if ($f $h)
            (cons-atom $h $rest)  ;; Include the head if it satisfies the predicate
            $rest  ;; Otherwise, skip the head
        )
      )
  )
)

(= (is-even $x) (== (% $x 2) 0))  ;; Predicate to check if a number is even

!(filter is-even (1 2 3 4))  ;; Example usage: Should return (2 4) s

( = ( bin ) 0)
(   =  (bin) 1)

!((superpose  bin) )
