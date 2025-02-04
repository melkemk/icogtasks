
(: List Type)
(: Nil List)
(: Cons (-> Number List List))

! (get-type (Cons 1 Nil)) ; List
! (Cons 1 (Cons 2 (Cons 3 Nil))) ; [1,2,3]

(= (pop (Cons $x Nil)) Nil)
(= (pop (Cons $x (Cons $y $ys))) (Cons $x (pop (Cons $y $ys))))
! (pop (Cons 1 (Cons 2 (Cons 3 Nil))))