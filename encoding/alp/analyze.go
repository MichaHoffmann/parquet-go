package alp

import (
	"math"
	"math/bits"
)

// analyzeResult is the output of one per-(exponent, factor) vector analysis:
// the number of exceptions and the minimum/maximum encoded value across the
// non-exception lanes. Its field layout is shared with the assembly kernels
// (analyze_amd64.s) and must not be reordered or repacked.
type analyzeResult struct {
	exceptions int64
	minEnc     int32
	maxEnc     int32
}

// analyzeFloatScalar is the scalar reference for the SIMD analyzers.
func analyzeFloatScalar(values []float32, exponent, factor int) (exceptions int, minEnc, maxEnc int32) {
	minEnc = math.MaxInt32
	maxEnc = math.MinInt32
	for i := range values {
		enc, exc := evalFloat(values[i], exponent, factor)
		if exc {
			exceptions++
		} else {
			if enc < minEnc {
				minEnc = enc
			}
			if enc > maxEnc {
				maxEnc = enc
			}
		}
	}
	return exceptions, minEnc, maxEnc
}

// analyzeFloatBoundedScalar stops once the candidate cannot beat maxLowerBound.
// The processed count remains available to verify pruning behavior.
func analyzeFloatBoundedScalar(values []float32, exponent, factor int, maxLowerBound int64) (exceptions int, minEnc, maxEnc int32, complete bool, processed int) {
	minEnc = math.MaxInt32
	maxEnc = math.MinInt32
	for i := range values {
		enc, exc := evalFloat(values[i], exponent, factor)
		if exc {
			exceptions++
		} else {
			if enc < minEnc {
				minEnc = enc
			}
			if enc > maxEnc {
				maxEnc = enc
			}
		}
		processed++
		if floatAnalysisLowerBound(len(values), exceptions, minEnc, maxEnc) > maxLowerBound {
			return exceptions, minEnc, maxEnc, false, processed
		}
	}
	return exceptions, minEnc, maxEnc, true, processed
}

func floatAnalysisLowerBound(length, exceptions int, minEnc, maxEnc int32) int64 {
	bitWidth := 0
	if minEnc <= maxEnc {
		delta := int64(maxEnc) - int64(minEnc)
		if delta != 0 {
			bitWidth = bits.Len64(uint64(delta))
		}
	}
	return int64(length)*int64(bitWidth) + int64(exceptions)*48
}

func analyzeDouble(values []float64, exponent, factor int) (int, int64, int64) {
	return analyzeDoubleScalar(values, exponent, factor)
}

// analyzeDoubleScalar is the scalar reference for future SIMD analyzers.
func analyzeDoubleScalar(values []float64, exponent, factor int) (exceptions int, minEnc, maxEnc int64) {
	minEnc = math.MaxInt64
	maxEnc = math.MinInt64
	for i := range values {
		enc, exc := evalDouble(values[i], exponent, factor)
		if exc {
			exceptions++
		} else {
			if enc < minEnc {
				minEnc = enc
			}
			if enc > maxEnc {
				maxEnc = enc
			}
		}
	}
	return exceptions, minEnc, maxEnc
}

func analyzeDoubleBounded(values []float64, exponent, factor int, maxLowerBound int64) (exceptions int, minEnc, maxEnc int64, complete bool, processed int) {
	return analyzeDoubleBoundedScalar(values, exponent, factor, maxLowerBound)
}

func analyzeDoubleBoundedScalar(values []float64, exponent, factor int, maxLowerBound int64) (exceptions int, minEnc, maxEnc int64, complete bool, processed int) {
	minEnc = math.MaxInt64
	maxEnc = math.MinInt64
	for i := range values {
		enc, exc := evalDouble(values[i], exponent, factor)
		if exc {
			exceptions++
		} else {
			if enc < minEnc {
				minEnc = enc
			}
			if enc > maxEnc {
				maxEnc = enc
			}
		}
		processed++
		if doubleAnalysisLowerBound(len(values), exceptions, minEnc, maxEnc) > maxLowerBound {
			return exceptions, minEnc, maxEnc, false, processed
		}
	}
	return exceptions, minEnc, maxEnc, true, processed
}

func doubleAnalysisLowerBound(length, exceptions int, minEnc, maxEnc int64) int64 {
	bitWidth := 0
	if minEnc <= maxEnc {
		delta := uint64(maxEnc) - uint64(minEnc)
		if delta != 0 {
			bitWidth = bits.Len64(delta)
		}
	}
	return int64(length)*int64(bitWidth) + int64(exceptions)*80
}

// encodeFloatDeltasScalar writes FOR deltas and compact exception positions.
// Exception lanes are zeroed here and normalized by the vector encoder once all
// positions are known.
//
// excPos must have exactly as many entries as evalFloat classifies as
// exceptions.
func encodeFloatDeltasScalar(values []float32, exponent, factor int, minEnc int32, deltas []uint32, excPos []uint16) {
	umin := uint32(minEnc)
	exceptionIndex := 0
	for i := range values {
		e, exc := evalFloat(values[i], exponent, factor)
		if exc {
			deltas[i] = 0
			excPos[exceptionIndex] = uint16(i)
			exceptionIndex++
		} else {
			deltas[i] = uint32(e) - umin
		}
	}
}

// analyzeFloatEncodedScalar also materializes encoded values and exception
// positions for the singleton-preset path.
func analyzeFloatEncodedScalar(values []float32, exponent, factor int, encoded []uint32, excPos []uint16) (exceptions int, minEnc, maxEnc int32) {
	minEnc = math.MaxInt32
	maxEnc = math.MinInt32
	for i := range values {
		enc, exc := evalFloat(values[i], exponent, factor)
		encoded[i] = uint32(enc)
		if exc {
			excPos[exceptions] = uint16(i)
			exceptions++
		} else {
			if enc < minEnc {
				minEnc = enc
			}
			if enc > maxEnc {
				maxEnc = enc
			}
		}
	}
	return exceptions, minEnc, maxEnc
}
