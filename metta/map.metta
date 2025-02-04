(: List Type)
(: Nil List)
(: Cons (-> Number List List))

! (get-type (Cons 1 Nil)) ; List
! (Cons 1 (Cons 2 (Cons 3 Nil))) ; [1,2,3]

(= (map $f Nil) Nil)

(= (map $f (Cons $x $xs))
    (Cons ($f $x) (map $f $xs))
)
(= (square $x) (* $x $x))

! (map square (Cons 1 (Cons 2 (Cons 3 Nil))))
