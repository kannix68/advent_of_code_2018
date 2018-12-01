#!/usr/bin/env clojure

; Advent of code 2018, AoC day 1 puzzle 1.
; This solution (clojure 1.8) by kannix68, @ 2018-12-01.

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
;(defn transpose-lol [lol]
;  "transpose rows <=> columns of a list-of-lists"
;  (apply map list lol)
;)
;(defn get-max-freq-colelems [loltr]
;  "get a list of the keys having max frequency for each sequenc in list-of-sequences."
;  (map #(key (apply min-key val (frequencies %))) loltr) ; use/map closure #() on each list-elem
;)

;** test data (as a var(iable))
(def teststr "+1
-2
+3
+1")

;** MAIN

; solve/assert test-data
(let [
  lst (map read-string (explode-str teststr))
  freqsum (apply + lst) ; apply forces coercion/reduction
  result freqsum
  expected 3
  ]
  (println "test-string\n" teststr)
  (println "list" lst)
  (println "freqsum" freqsum)
  (assert-msg (= result expected) (str "test-result " result " should be " expected))
)

; solve my specific data input
(let [
  datastr (slurp "day01.in")
  lst (map read-string (explode-str datastr))
  freqsum (apply + lst) ; apply forces coercion/reduction
  result freqsum
  ]
  ;(println (str "input   #" collen)  (take 2 lol)   "..." (take-last 1 lol))
  ;(println (str "transpd #" linelen) (take 2 loltr) "..." (take-last 1 loltr))
  ;(println "data line-len=" linelen ", col-len=" collen)
  (println "result :=> " result)
)
