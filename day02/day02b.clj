#!/usr/bin/env clojure

; Advent of code 2018, AoC day 2 part 1.
; This solution (clojure 1.8) by kannix68, @ 2018-12-02.
; Save your specific input as "day##.in".
;

(require '[clojure.string :as str]) ; clj str ops

;** test data (as a var(iable))
(def teststr1 "abcdef
bababc
abbcde
abcccd
aabcdd
abcdee
ababab")

;** "lib" functions

(defn assert-msg [condit msg]
  "check condition, fail with message on false, print message otherwise"
  (if (not condit)
    (assert condit (str "assert-ERROR:" msg))
    (println "assert-OK:" msg))
)

;** problem solution
(defn explode-str-lol [str]
  "explode a (multiline) string into list-of-lists,
   containing single characters grouped by line."
  (map #(seq (str/split % #"")) (str/split str #"\r?\n"))
)

(defn count-occur-in [lol v]
  "take a list of lists, count number of lists containing value v"
  (let [
    freqs (map frequencies lol)  ; list-of-lists
    valz (map vals freqs)  ; list-of-lists
    fltrdvalz (filter (fn[lst] (some (fn[it] (= v it)) lst)) valz)
    ]
    ;(println "freqs:" freqs)
    ;(println "valz:" valz)
    ;(println "fltrdvalz:" fltrdvalz)
    (count fltrdvalz)
  )
)

;** MAIN

; solve
(defn solve [input]
  (let [
    ;deb 12 ; max take slice items
    lol (explode-str-lol input)
    count2s (count-occur-in lol 2)
    count3s (count-occur-in lol 3)
    result 0
    ]
    ;(println "input\n" input)
    (println "list-count" (count lol)) ;"list-of-lists" lol)
    (println "count-occur-in lol 2:" count2s)
    (println "count-occur-in lol 3:" count3s)
    (* count2s count3s)
  )
)

; solve/assert test-data
(let [
  expected 12
  result (solve teststr1)
  ]
  (assert-msg (= result expected) (str "test-result " result " should be " expected))
)

; solve my specific data input
(let [
  datastr (slurp "day02.in")
  result (solve datastr)
  ]
  (println "result :=>" result)
)
