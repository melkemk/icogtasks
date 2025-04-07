; function: includes?
; task: checks if a list contains a specific atom

(: isMember? (-> Expression Atom boolean))
(= (isMember? $list $element)
  (let $foundElement (
    collapse (
      let $currentElement (superpose $list)
      (if (== $currentElement $element)
        $currentElement
        (empty)
      )
    )
  )
  (if (== $foundElement ())
    False
    True
  ))
)
; Example usage:  
!(isMember? (1 2 3 4 5) 3) ;; Should output: True
!(isMember? (1 2 3 4 5) 6) ;; Should output: False
; function: isMember