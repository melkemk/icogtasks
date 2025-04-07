(= (zip $xs $ys)
  (if (or (== $xs ()) (== $ys ()))
      ()  ;; base case: one is empty
      (let* (
          (($x $xs-rest) (decons-atom $xs))  ;; Deconstruct the first list
          (($y $ys-rest) (decons-atom $ys))  ;; Deconstruct the second list
          ($rest (zip $xs-rest $ys-rest))   ;; Recursively zip the rest
        )
        (cons-atom ($x $y) $rest)  ;; Combine the current pair with the rest
      )
  )
)

!(zip (1 2 5 5) (3 4 5 5 6))  ;; Example usage