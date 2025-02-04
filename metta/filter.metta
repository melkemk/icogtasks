(: List Type)
(: Nil List)
(: Cons (-> Number List List))

(= (filter $f Nil) Nil)
(= (filter $f (Cons $x $xs))
    (if ($f $x)
        (Cons $x (filter $f $xs))
        (filter $f $xs)))
(= (is-even $x) (== (% $x 2) 0))

! (filter is-even (Cons 1 (Cons 2 (Cons 3 (Cons 4 Nil)))))