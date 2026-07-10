//go:build purego || !amd64

package alp

// analyzeFloat uses the scalar implementation when no SIMD kernel is built.
func analyzeFloat(values []float32, exponent, factor int) (int, int32, int32) {
	return analyzeFloatScalar(values, exponent, factor)
}

func analyzeFloatBounded(values []float32, exponent, factor int, maxLowerBound int64) (exceptions int, minEnc, maxEnc int32, complete bool, processed int) {
	return analyzeFloatBoundedScalar(values, exponent, factor, maxLowerBound)
}

// encodeFloatDeltas falls back to the portable scalar implementation.
func encodeFloatDeltas(values []float32, exponent, factor int, minEnc int32, deltas []uint32, excPos []uint16) {
	encodeFloatDeltasScalar(values, exponent, factor, minEnc, deltas, excPos)
}

func analyzeFloatEncoded(values []float32, exponent, factor int, encoded []uint32, excPos []uint16) (int, int32, int32) {
	return analyzeFloatEncodedScalar(values, exponent, factor, encoded, excPos)
}

const useEncodeAVX2 = false

const useAnalyzeAVX2 = false
