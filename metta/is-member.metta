(: List Type)
(: Nil List)
(: Cons (-> Number List List))

(= (is-member $x  Nil) false)  , return false.

(= (is-member $x (Cons $y $xs))   (if (== $x $y ) true (is-member $x $xs) ))
! (is-member 2 (Cons 1 (Cons 2 (Cons 3 Nil))))
! (is-member 0 (Cons 1 (Cons 2 (Cons 3 Nil))))
! (is-member 3 (Cons 1 (Cons 2 (Cons 3 Nil))))