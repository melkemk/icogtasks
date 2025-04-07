; function:  findAndRemove
; task: remove every match of a specific atom from the list

(:  findAndRemove (-> Expression Atom Expression))
(= ( findAndRemove $list $target)
    (collapse
        (let $unit (superpose $list)
            (if (== $unit $target)
                (empty)
                $unit
            )
        )
    )
)

; Example usage:  
!(findAndRemove (1 2 3 4 5) 3) ;; Should output: (1 2 4 5)