//go:build !purego

package alp

import (
	"math"

	"golang.org/x/sys/cpu"
)

// analyzeBoundResult is shared with analyze_bounded_amd64.s.
type analyzeBoundResult struct {
	exceptions int64
	minEnc     int32
	maxEnc     int32
	processed  int64
	pruned     int64
}

// analyzeFloatAVX2 processes len(values) float32 values (which the caller
// guarantees to be a non-zero multiple of 8) for a fixed set of precomputed
// scaling factors, writing the exception count and min/max encoded value into
// res. res.minEnc / res.maxEnc must be pre-seeded (to MaxInt32 / MinInt32) so
// the caller can merge a scalar-processed tail. It is bit-exactly equivalent to
// analyzeFloatScalar.
//
//go:noescape
func analyzeFloatAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, res *analyzeResult)

//go:noescape
func analyzeFloatBoundedAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, maxLowerBound int64, res *analyzeBoundResult)

// encodeFloatAVX2 is the encode-pass counterpart to analyzeFloatAVX2: for the
// caller's precomputed scaling factors it writes each lane's FOR delta
// (encoded-minEnc) into deltas and writes exception positions in ascending order
// to excPos. Exception placeholders are normalized by the vector encoder after
// this kernel returns. len(values) must be a non-zero multiple of 8; deltas must
// hold len(values) uint32 and excPos exactly the number of exceptions in values.
// It is bit-exactly equivalent to encodeFloatDeltasScalar over the same lanes.
//
//go:noescape
func encodeFloatAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, minEnc int32, deltas []uint32, excPos []uint16)

// analyzeFloatEncodedAVX2 additionally retains each raw encoded lane and
// compacts exception positions into excPos. len(values) must be a non-zero
// multiple of 8.
//
//go:noescape
func analyzeFloatEncodedAVX2(values []float32, pe, nf, pf, ne, magic, upper float32, encoded []uint32, excPos []uint16, res *analyzeResult)

var useEncodeAVX2 = cpu.X86.HasAVX2

// useAnalyzeAVX2 controls analyzer kernels that execute POPCNT in addition to
// AVX2 instructions.
var useAnalyzeAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasPOPCNT

// analyzeFloat dispatches to the AVX2 kernel for the bulk of the vector and
// finishes any sub-8 remainder with the scalar path, falling back entirely to
// scalar when AVX2 is unavailable.
func analyzeFloat(values []float32, exponent, factor int) (int, int32, int32) {
	if useAnalyzeAVX2 && len(values) >= 8 {
		n8 := len(values) &^ 7
		res := analyzeResult{minEnc: math.MaxInt32, maxEnc: math.MinInt32}
		analyzeFloatAVX2(values[:n8],
			floatPow10[exponent], floatPow10Negative[factor],
			floatPow10[factor], floatPow10Negative[exponent],
			magicFloat, floatEncodingUpperLimit, &res)
		exceptions := int(res.exceptions)
		minEnc, maxEnc := res.minEnc, res.maxEnc
		for i := n8; i < len(values); i++ {
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
	return analyzeFloatScalar(values, exponent, factor)
}

func analyzeFloatBounded(values []float32, exponent, factor int, maxLowerBound int64) (exceptions int, minEnc, maxEnc int32, complete bool, processed int) {
	if !useAnalyzeAVX2 || len(values) < 8 {
		return analyzeFloatBoundedScalar(values, exponent, factor, maxLowerBound)
	}

	n8 := len(values) &^ 7
	res := analyzeBoundResult{minEnc: math.MaxInt32, maxEnc: math.MinInt32}
	analyzeFloatBoundedAVX2(values[:n8],
		floatPow10[exponent], floatPow10Negative[factor],
		floatPow10[factor], floatPow10Negative[exponent],
		magicFloat, floatEncodingUpperLimit, maxLowerBound, &res)
	exceptions, minEnc, maxEnc = int(res.exceptions), res.minEnc, res.maxEnc
	processed = int(res.processed)
	if res.pruned != 0 {
		return exceptions, minEnc, maxEnc, false, processed
	}

	for i := n8; i < len(values); i++ {
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

// encodeFloatDeltas dispatches the encode pass to the AVX2 kernel for the bulk
// of the vector (whole 8-lane groups) and finishes any sub-8 remainder with the
// scalar path, falling back entirely to scalar when AVX2 is unavailable.
func encodeFloatDeltas(values []float32, exponent, factor int, minEnc int32, deltas []uint32, excPos []uint16) {
	if useEncodeAVX2 && len(values) >= 8 {
		n8 := len(values) &^ 7
		var tailPos [7]uint16
		tailExceptions := 0
		umin := uint32(minEnc)
		for i := n8; i < len(values); i++ {
			e, exc := evalFloat(values[i], exponent, factor)
			if exc {
				deltas[i] = 0
				tailPos[tailExceptions] = uint16(i)
				tailExceptions++
			} else {
				deltas[i] = uint32(e) - umin
			}
		}
		prefixExceptions := len(excPos) - tailExceptions
		encodeFloatAVX2(values[:n8],
			floatPow10[exponent], floatPow10Negative[factor],
			floatPow10[factor], floatPow10Negative[exponent],
			magicFloat, floatEncodingUpperLimit,
			minEnc, deltas[:n8], excPos[:prefixExceptions])
		copy(excPos[prefixExceptions:], tailPos[:tailExceptions])
		return
	}
	encodeFloatDeltasScalar(values, exponent, factor, minEnc, deltas, excPos)
}

// analyzeFloatEncoded performs the singleton-preset analysis while retaining
// the raw encoded lanes and compact exception positions for the packing pass.
func analyzeFloatEncoded(values []float32, exponent, factor int, encoded []uint32, excPos []uint16) (int, int32, int32) {
	if useAnalyzeAVX2 && len(values) >= 8 {
		n8 := len(values) &^ 7
		var tailPos [7]uint16
		tailExceptions := 0
		tailMin, tailMax := int32(math.MaxInt32), int32(math.MinInt32)
		for i := n8; i < len(values); i++ {
			enc, exc := evalFloat(values[i], exponent, factor)
			encoded[i] = uint32(enc)
			if exc {
				tailPos[tailExceptions] = uint16(i)
				tailExceptions++
			} else {
				if enc < tailMin {
					tailMin = enc
				}
				if enc > tailMax {
					tailMax = enc
				}
			}
		}
		res := analyzeResult{minEnc: math.MaxInt32, maxEnc: math.MinInt32}
		prefixExceptions := len(excPos) - tailExceptions
		analyzeFloatEncodedAVX2(values[:n8],
			floatPow10[exponent], floatPow10Negative[factor],
			floatPow10[factor], floatPow10Negative[exponent],
			magicFloat, floatEncodingUpperLimit,
			encoded[:n8], excPos[:prefixExceptions], &res)
		exceptions := int(res.exceptions) + tailExceptions
		minEnc, maxEnc := res.minEnc, res.maxEnc
		if tailMin < minEnc {
			minEnc = tailMin
		}
		if tailMax > maxEnc {
			maxEnc = tailMax
		}
		copy(excPos[exceptions-tailExceptions:exceptions], tailPos[:tailExceptions])
		return exceptions, minEnc, maxEnc
	}
	return analyzeFloatEncodedScalar(values, exponent, factor, encoded, excPos)
}
