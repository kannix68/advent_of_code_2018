#!/usr/bin/env clojure

; Advent of code 2018, AoC day 2 part 2.
; This solution (clojure 1.8) by kannix68, @ 2018-12-02.
; Save your specific input as "day##.in".
;
; tips:
; * [diff - clojure.data](https://clojuredocs.org/clojure.data/diff)
; * [combinatorics - Enumerate k-combinations in Clojure (P26 from 99 problems)](<https://codereview.stackexchange.com/questions/8930/enumerate-k-combinations-in-clojure-p26-from-99-problems>)
;     * [H-99: Ninety-Nine Haskell Problems](https://wiki.haskell.org/H-99:_Ninety-Nine_Haskell_Problems)
;

(require '[clojure.string :as str]) ; clj str ops
(require '[clojure.data :as dat])
(require '[clojure.set :as set])
;(require '[clojure.math.combinatorics :as combi]) ; non-core dependency

(defn assert-msg [condit msg]
  "check condition, fail with message on false, print message otherwise"
  (if (not condit)
    (assert condit (str "assert-ERROR:" msg))
    (println "assert-OK:" msg))
)

;** problem solution
(defn explode-str [str]
  "explode a (singleline) string into a listof characters."
  (str/split str #"")
)

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

(defn combinations [s k]
  "Given a set S and a no. of items K,
  returns all possible combinations of K items that can be taken from set S."
  (cond
    (> k (count s)) nil    ;not enough items in sequence to form a valid combination
    (= k (count s)) [s]    ;only one combination available: all items 
    (= 1 k) (map vector s) ;every item (on its own) is a valid combination
    :else (reduce concat (map-indexed 
            (fn [i x] (map #(cons x %) (combinations (drop (inc i) s) (dec k)))) 
            s))))

;(defn str-dist [s1 s2]
(defn lst-dist [l1 l2]
  "how many items in lists are different?, returniing int"
  (count (filter identity (first (dat/diff l1 l2)))) ; filter identity removes nils
)

;** test data (as a var(iable))
(def teststr1 "abcde
fghij
klmno
pqrst
fguij
axcye
wvxyz")

;** MAIN

; solve
(defn solve [input]
  (let [
    lol (explode-str-lol input)
    ;lstdst (lst-dist (explode-str "ab") (explode-str "ac"))
    ;lstdst2 (lst-dist (explode-str "abc") (explode-str "cba"))
    ;lstdst3 (lst-dist (explode-str "abcde") (explode-str "axcye"))
    ;lstdst4 (lst-dist (explode-str "fghij") (explode-str "fguij"))
    ;combis (combi/combinations lol 2)
    combis (combinations lol 2)
    fltrdlst (apply concat (filter (fn[lst] (= 1 (apply lst-dist lst))) combis)) ; apply concat unwrap one level
    intrsect (set/intersection (set(first fltrdlst)) (set(last fltrdlst)))
    ;dsts (map lst-dist combis)
    result (str/join (filter #(contains? intrsect %) (first fltrdlst))) ; in original order!
    ]
    ;(println "input\n" input)
    (println "list-count" (count lol)) ;"list-of-lists" lol)
    ;(println "lstdst:" lstdst)
    ;(println "lstdst2:" lstdst2)
    ;(println "lstdst3:" lstdst3)
    ;(println "lstdst4:" lstdst4)
    ;(println "combis:" combis)
    ;(println "fltrdlst:" fltrdlst)
    ;(println "intrsect:" intrsect)
    (println "result:" result)
    ;(println "dsts:" dsts)
    result
  )
)

; solve/assert test-data
(let [
  expected "fgij"
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
