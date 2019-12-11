#!/usr/bin/env clojure

; Advent of code 2018, AoC day 1 puzzle 2.
; This solution (clojure 1.8) by kannix68, @ 2018-12-01.
; Save your specific input as "day##.in".
;
; tips:
; * [clojure - Returning duplicates in a sequence](<https://stackoverflow.com/questions/8056645/returning-duplicates-in-a-sequence>)
; * find-first: [clojure - How to stop iterating a sequence when a condition is met](https://stackoverflow.com/questions/11866446/how-to-stop-iterating-a-sequence-when-a-condition-is-met)
;     * [find first match in a sequence](<https://groups.google.com/forum/#!topic/clojure/ojh-3_VXoac>)
; * could be useful: [reduced - clojure.core](<https://clojuredocs.org/clojure.core/reduced>)

(require '[clojure.string :as str]) ; clj str ops

(defn assert-msg [condit msg]
  "check condition, fail with message on false, print message otherwise"
  (if (not condit)
    (assert condit (str "assert-ERROR:" msg))
    (println "assert-OK:" msg))
)

;** problem solution
(defn explode-str [str]
  "explode a (multiline) string into list-of-ints,
   by splitting on newlines, each line giving an integer."
  ;(map #(seq (str/split % #"")) (str/split str #"\r?\n"))
  (str/split str #"\r?\n")
)

;** test data (as a var(iable))
(def teststr1 "+1
-2
+3
+1")

(def teststr2 "+1
-1")

(def teststr3 "+3
+3
+4
-2
-4")

;** MAIN
; frequencies is NOT lazy

;* ; NOT lazy
; (defn dups [seq]
;   (for [[id freq] (frequencies seq)  ;; get the frequencies, destructure
;         :when (> freq 1)]            ;; this is the filter condition
;    id))                              ;; just need the id, not the frequency

; (defn get-cycle [xs]
;   (first (filter #(number? (first %))
;     (reductions
;       (fn [[m i] x] (if-let [xat (m x)] [xat i] [(assoc m x i) (inc i)]))
;       [(hash-map) 0] xs))))

(defn first-dup [coll]
  (reduce (fn [acc [idx x]]
            (if-let [v (get acc x)]
              (reduced (conj v idx))
              (assoc acc x [idx])))
          {} (map-indexed #(vector % %2) coll)))

; solve
(defn solve [input]
  (let [
    ;deb 12 ; max take slice items
    lst (map read-string (explode-str input))
    freqsum (apply + lst) ; apply forces coercion/reduction
    lstcycle (conj (cycle lst) 0)
    cumsumlst (reductions + lstcycle) ; running sums as sequence
    fidx (first (first-dup cumsumlst)) ; use defined first-dup fn
    fval (nth cumsumlst fidx)
    result fval
    ]
    ;(println "test-string\n" teststr)
    ;(println "list" lst "list-count" (count (doall lst)))
    ;(println (take deb lstcycle))
    ;(println (take deb cumsumlst))
    ;(println "idx:" fidx "value:" fval)
    result
  )
)

; solve/assert test-data
(let [
  expected 2
  result (solve teststr1)
  ]
  (assert-msg (= result expected) (str "test-result " result " should be " expected))
)
(let [
  expected 0
  result (solve teststr2)
  ]
  (assert-msg (= result expected) (str "test-result " result " should be " expected))
)
(let [
  expected 10
  result (solve teststr3)
  ]
  (assert-msg (= result expected) (str "test-result " result " should be " expected))
)

; solve my specific data input
(let [
  datastr (slurp "day01.in")
  result (solve datastr)
  ]
  (println "result :=>" result)
)
