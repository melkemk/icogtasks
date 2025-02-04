(: List Type)
(: Nil List)
(: Cons (-> Number List List))

! (get-type (Cons 1 Nil)) ; List
! (Cons 1 (Cons 2 (Cons 3 Nil))) ; [1,2,3]

(= (max-value (Cons $x Nil)) $x)

(= (max-value  (Cons $x (Cons $y $ys)))
    (max-value (Cons (if (> $x $y) $x $y) $ys)))

! (max-value (Cons 4 (Cons 1 (Cons 7 (Cons 3 Nil)))))
! (max-value (Cons 4 (Cons 1 (Cons 7 (Cons 3 Nil)))))
! (max-value (Cons 9 (Cons 2 (Cons 5 Nil))))
! (max-value (Cons 3 (Cons 3 (Cons 3 Nil))))