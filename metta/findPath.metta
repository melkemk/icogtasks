(NewYork LosAngeles)
(NewYork Chicago)
(NewYork Miami)
(LosAngeles SanFrancisco)
(Chicago Houston)
(Houston Dallas)
(Dallas Denver)
(Denver Seattle)
(Denver amx)
(amx Seattle)
(Seattle Portland) 

(= (find_path $start $end $path)
  (match &self ($start $x)
    (if (== $x $end)
      (cons-atom $path ($x))  ;; Base case: reached the destination
      (let* ( 
        ($new_path (cons-atom $path ($x)))  ;; Add the current node to the path
      )
      (find_path $x $end $new_path)  ;; Recursively search for the path
      )
    )
  )
)

!(find_path NewYork Seattle ())