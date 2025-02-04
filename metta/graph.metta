 ; Implementation to reverse all matching edges in the atom space, not just a single pattern match
 ; Define initial graph
(edge A B)
(edge B C)
(edge C D)
(edge D A)

 ; Define a rule to reverse edges
(= (reverse-edge)
    (match &self (edge $x $y)
        (let ()
            (remove-atom &self (edge $x $y))
            (add-atom &self (edge $y $x)))))

 ; Function to print all edges
(= (print-edges)
    (match &self (edge $x $y)
        (edge $x $y)))

 ; Print initial graph
! (println! "Initial graph:")
! (print-edges)

! (reverse-edge)

 ; Print the modified graph
! (print-edges)
! (println! "Graph after reversing edges:")

