(: zipWith (-> Expression Expression (-> Number Number Number) Expression))
(= (zipWith $xs $ys $function)
   (if (or (== $xs ()) (== $ys ()))
       ()  ;; Base case: either list empty, return empty
       (let* (
             (($x $xs-rest) (decons-atom $xs))   ;; Deconstruct $xs
             (($y $ys-rest) (decons-atom $ys))   ;; Deconstruct $ys
             ($rest (zipWith $xs-rest $ys-rest $function)))  ;; Recurse
             (cons-atom ($function $x $y) $rest))))  ;; Apply function and build list

;; Example usage:
!(zipWith (1 2 5 5) (3 4 5 5 6) +)  ;; Should output: (4 6 10 10)