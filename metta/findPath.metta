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

(= (find_path $start $end $reverse_path)
  (if (== $start $end)
      (reverse $reverse_path (Cons $start Nil))  ;; Reverse accumulated path
      (match &self ($start $x)
        (find_path $x $end (cons-atom $start $reverse_path))))  ;; Prepend $start
)

(= (reverse $list $acc)
  (if (== $list ())
      $acc
      (let* ((($h $t) (decons-atom $list)))
            (reverse $t (cons-atom $h $acc)))))

!(find_path NewYork Seattle ())