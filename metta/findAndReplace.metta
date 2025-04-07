; function: swapIn
; task: replaces every occurrence of one atom with another

(: swapIn (-> Expression Atom Atom Expression))
(= (swapIn $list $val $replacement)
    (collapse
        (let $elem (superpose $list)
            (if (== $elem $val)
                $replacement
                $elem
            )
        )
    )
)
; Example usage:
!(swapIn (1 2 3 4 5) 3 99) 
;; Should output: (1 2 99 4 5)