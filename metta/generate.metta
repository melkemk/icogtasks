(: generate (-> Number Number Number Expression))

(= (generate $start $end $step)
  (if (> $start $end)
      ()
      (let $rest (generate (+ $start $step) $end $step)
        (cons-atom $start $rest)
      )
  )
)
!(generate 1 10 2) ;; => (1 3 5 7 9)
!(generate 5 20 3) ;; => (5 8 11 14 17 20)
!(generate 10 10 1) ;; => (10)