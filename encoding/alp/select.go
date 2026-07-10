package alp

import (
	"math"
)

var (
	allValidFloatPairs  = buildValidPairs(floatMaxExponent)
	allValidDoublePairs = buildValidPairs(doubleMaxExponent)
)

func buildValidPairs(maxExponent int) [][2]int {
	pairs := make([][2]int, 0, (maxExponent+1)*(maxExponent+2)/2)
	for e := 0; e <= maxExponent; e++ {
		for f := 0; f <= e; f++ {
			pairs = append(pairs, [2]int{e, f})
		}
	}
	return pairs
}

// topPresets returns the most frequent pairs in deterministic rank order.
func topPresets(counts map[[2]int]int) [][2]int {
	combos := make([]presetCount, 0, len(counts))
	for k, c := range counts {
		combos = append(combos, presetCount{pair: k, count: c})
	}
	return rankPresets(make([][2]int, maxPresetCombinations), combos)
}

type presetCount struct {
	pair  [2]int
	count int
}

func countPreset(counts []presetCount, pair [2]int) []presetCount {
	for i := range counts {
		if counts[i].pair == pair {
			counts[i].count++
			return counts
		}
	}
	return append(counts, presetCount{pair: pair, count: 1})
}

func rankPresets(dst [][2]int, counts []presetCount) [][2]int {
	for i := 1; i < len(counts); i++ {
		for j := i; j > 0 && presetRanksBefore(counts[j], counts[j-1]); j-- {
			counts[j], counts[j-1] = counts[j-1], counts[j]
		}
	}
	n := min(len(counts), maxPresetCombinations)
	dst = dst[:n]
	for i := range n {
		dst[i] = counts[i].pair
	}
	return dst
}

func presetRanksBefore(a, b presetCount) bool {
	if a.count != b.count {
		return a.count > b.count
	}
	if a.pair[0] != b.pair[0] {
		return a.pair[0] < b.pair[0]
	}
	return a.pair[1] < b.pair[1]
}

func sampleVectorIndex(i, samples, numVectors int) int {
	return i * (numVectors - 1) / (samples - 1)
}

func findBestFloatParams(values []float32) (exponent, factor, numExceptions int, minEnc, maxEnc int32) {
	return pickBestFloatBounded(values, 0, len(values), allValidFloatPairs)
}

func findBestFloatParamsWithPresets(values []float32, offset, length int, presets [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int32) {
	return pickBestFloat(values, offset, length, presets)
}

// pickBestFloatBounded applies exact branch-and-bound to full-grid searches.
func pickBestFloatBounded(values []float32, offset, length int, pairs [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int32) {
	bestExponent := pairs[0][0]
	bestFactor := pairs[0][1]
	bestExceptions := length
	var bestMin, bestMax int32
	bestEstimatedSize := int64(math.MaxInt64)

	window := values[offset : offset+length]
	for _, pair := range pairs {
		e := pair[0]
		f := pair[1]
		exceptions, minEncoded, maxEncoded, complete, _ := analyzeFloatBounded(window, e, f, bestEstimatedSize)
		if !complete || exceptions == length {
			continue
		}
		estimatedSize := floatAnalysisLowerBound(length, exceptions, minEncoded, maxEncoded)
		if estimatedSize < bestEstimatedSize ||
			(estimatedSize == bestEstimatedSize &&
				(e > bestExponent || (e == bestExponent && f > bestFactor))) {
			bestEstimatedSize = estimatedSize
			bestExponent = e
			bestFactor = f
			bestExceptions = exceptions
			bestMin = minEncoded
			bestMax = maxEncoded
			if bestExceptions == 0 && minEncoded == maxEncoded {
				return bestExponent, bestFactor, 0, minEncoded, maxEncoded
			}
		}
	}
	return bestExponent, bestFactor, bestExceptions, bestMin, bestMax
}

// pickBestFloat scores pairs by packed width and exception cost.
func pickBestFloat(values []float32, offset, length int, pairs [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int32) {
	bestExponent := pairs[0][0]
	bestFactor := pairs[0][1]
	bestExceptions := length
	var bestMin, bestMax int32
	bestEstimatedSize := int64(math.MaxInt64)

	window := values[offset : offset+length]
	for _, pair := range pairs {
		e, f := pair[0], pair[1]
		exceptions, minEncoded, maxEncoded := analyzeFloat(window, e, f)
		if exceptions == length {
			continue
		}
		estimatedSize := floatAnalysisLowerBound(length, exceptions, minEncoded, maxEncoded)
		if estimatedSize < bestEstimatedSize ||
			(estimatedSize == bestEstimatedSize &&
				(e > bestExponent || (e == bestExponent && f > bestFactor))) {
			bestEstimatedSize = estimatedSize
			bestExponent = e
			bestFactor = f
			bestExceptions = exceptions
			bestMin = minEncoded
			bestMax = maxEncoded
			if bestExceptions == 0 && minEncoded == maxEncoded {
				return bestExponent, bestFactor, 0, minEncoded, maxEncoded
			}
		}
	}
	return bestExponent, bestFactor, bestExceptions, bestMin, bestMax
}

func findBestDoubleParams(values []float64) (exponent, factor, numExceptions int, minEnc, maxEnc int64) {
	return pickBestDoubleBounded(values, 0, len(values), allValidDoublePairs)
}

func findBestDoubleParamsWithPresets(values []float64, offset, length int, presets [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int64) {
	return pickBestDouble(values, offset, length, presets)
}

// pickBestDoubleBounded is the double counterpart to pickBestFloatBounded.
func pickBestDoubleBounded(values []float64, offset, length int, pairs [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int64) {
	bestExponent := pairs[0][0]
	bestFactor := pairs[0][1]
	bestExceptions := length
	var bestMin, bestMax int64
	bestEstimatedSize := int64(math.MaxInt64)

	window := values[offset : offset+length]
	for _, pair := range pairs {
		e := pair[0]
		f := pair[1]
		exceptions, minEncoded, maxEncoded, complete, _ := analyzeDoubleBounded(window, e, f, bestEstimatedSize)
		if !complete || exceptions == length {
			continue
		}
		estimatedSize := doubleAnalysisLowerBound(length, exceptions, minEncoded, maxEncoded)
		if estimatedSize < bestEstimatedSize ||
			(estimatedSize == bestEstimatedSize &&
				(e > bestExponent || (e == bestExponent && f > bestFactor))) {
			bestEstimatedSize = estimatedSize
			bestExponent = e
			bestFactor = f
			bestExceptions = exceptions
			bestMin = minEncoded
			bestMax = maxEncoded
			if bestExceptions == 0 && minEncoded == maxEncoded {
				return bestExponent, bestFactor, 0, minEncoded, maxEncoded
			}
		}
	}
	return bestExponent, bestFactor, bestExceptions, bestMin, bestMax
}

func pickBestDouble(values []float64, offset, length int, pairs [][2]int) (exponent, factor, numExceptions int, minEnc, maxEnc int64) {
	bestExponent := pairs[0][0]
	bestFactor := pairs[0][1]
	bestExceptions := length
	var bestMin, bestMax int64
	bestEstimatedSize := int64(math.MaxInt64)

	window := values[offset : offset+length]
	for _, pair := range pairs {
		e := pair[0]
		f := pair[1]
		exceptions, minEncoded, maxEncoded := analyzeDouble(window, e, f)
		if exceptions == length {
			continue
		}
		estimatedSize := doubleAnalysisLowerBound(length, exceptions, minEncoded, maxEncoded)
		if estimatedSize < bestEstimatedSize ||
			(estimatedSize == bestEstimatedSize &&
				(e > bestExponent || (e == bestExponent && f > bestFactor))) {
			bestEstimatedSize = estimatedSize
			bestExponent = e
			bestFactor = f
			bestExceptions = exceptions
			bestMin = minEncoded
			bestMax = maxEncoded
			if bestExceptions == 0 && minEncoded == maxEncoded {
				return bestExponent, bestFactor, 0, minEncoded, maxEncoded
			}
		}
	}
	return bestExponent, bestFactor, bestExceptions, bestMin, bestMax
}
