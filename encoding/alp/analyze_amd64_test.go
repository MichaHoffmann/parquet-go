//go:build !purego

package alp

import (
	"math"
	"math/rand"
	"testing"
)

// callAVX2 seeds a result and runs the AVX2 kernel on a length-multiple-of-8
// slice, returning the same triple as analyzeFloatScalar.
func callAVX2(values []float32, e, f int) (int, int32, int32) {
	res := analyzeResult{minEnc: math.MaxInt32, maxEnc: math.MinInt32}
	analyzeFloatAVX2(values,
		floatPow10[e], floatPow10Negative[f],
		floatPow10[f], floatPow10Negative[e],
		magicFloat, floatEncodingUpperLimit, &res)
	return int(res.exceptions), res.minEnc, res.maxEnc
}

var analyzeFloatSpecialValues = []float32{
	0, -0, 1, -1, 0.5, -0.5, 0.1, -0.1, 123.456, -999.999,
	math.Float32frombits(0x7FC00001), math.Float32frombits(0xFFC00001),
	float32(math.Inf(1)), float32(math.Inf(-1)),
	float32(math.MaxFloat32), -float32(math.MaxFloat32),
	1e30, -1e30, 1e-30, math.Float32frombits(1), math.Float32frombits(0x80000001),
	12345678.0, 0.0009765625,
	math.Nextafter32(floatEncodingUpperLimit, 0), floatEncodingUpperLimit,
	math.Nextafter32(floatEncodingUpperLimit, float32(math.Inf(1))),
	math.Nextafter32(floatEncodingLowerLimit, 0), floatEncodingLowerLimit,
	math.Nextafter32(floatEncodingLowerLimit, float32(math.Inf(-1))),
}

func makeAnalyzeFloatValues(rng *rand.Rand, n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		switch rng.Intn(4) {
		case 0:
			values[i] = float32(rng.Intn(200000)) * 0.01
		case 1:
			values[i] = float32(rng.NormFloat64() * 1000)
		case 2:
			values[i] = analyzeFloatSpecialValues[rng.Intn(len(analyzeFloatSpecialValues))]
		default:
			values[i] = float32(rng.Intn(1000)) - 500
		}
	}
	return values
}

// TestAnalyzeFloatAVX2Differential asserts the AVX2 kernel is bit-exactly
// equivalent to the scalar reference for every valid (exponent, factor) pair
// across a wide range of inputs, including exceptional values.
func TestAnalyzeFloatAVX2Differential(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	rng := rand.New(rand.NewSource(1))
	vecs := [][]float32{
		analyzeFloatSpecialValues[:16],
		make([]float32, 1024),
	}
	for i := range vecs[1] {
		vecs[1][i] = analyzeFloatSpecialValues[10+i%2]
	}
	for _, n := range []int{8, 16, 64, 1024, 2048} {
		vecs = append(vecs, makeAnalyzeFloatValues(rng, n))
	}

	for vi, vec := range vecs {
		for _, pair := range allValidFloatPairs {
			e, f := pair[0], pair[1]
			we, wmin, wmax := analyzeFloatScalar(vec, e, f)
			ge, gmin, gmax := callAVX2(vec, e, f)
			if ge != we || gmin != wmin || gmax != wmax {
				t.Fatalf("vec %d (e=%d f=%d): AVX2=(exc %d,min %d,max %d) scalar=(exc %d,min %d,max %d)",
					vi, e, f, ge, gmin, gmax, we, wmin, wmax)
			}
		}
	}
}

// TestEncodeFloatAVX2Differential asserts both encoded-analysis and the regular
// encode pass agree with the scalar references over all valid pairs and lengths
// including non-multiples of eight.
func TestEncodeFloatAVX2Differential(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	rng := rand.New(rand.NewSource(2))
	var vecs [][]float32
	// Lengths include non-multiples of 8 to exercise the scalar remainder.
	for _, n := range []int{1, 7, 8, 9, 13, 16, 63, 64, 1000, 1023, 1024} {
		vecs = append(vecs, makeAnalyzeFloatValues(rng, n))
	}
	for _, n := range []int{1024, 1031} {
		v := make([]float32, n)
		for i := range v {
			v[i] = analyzeFloatSpecialValues[10+i%2]
		}
		vecs = append(vecs, v)
	}

	for vi, vec := range vecs {
		n := len(vec)
		refDeltas := make([]uint32, n)
		gotDeltas := make([]uint32, n)
		encoded := make([]uint32, n)
		for _, pair := range allValidFloatPairs {
			e, f := pair[0], pair[1]
			exceptions, minEnc, maxEnc := analyzeFloatScalar(vec, e, f)
			refPos := make([]uint16, exceptions)
			gotPos := make([]uint16, exceptions)
			encodedPos := make([]uint16, exceptions)
			gotExceptions, gotMin, gotMax := analyzeFloatEncoded(vec, e, f, encoded, encodedPos)
			if gotExceptions != exceptions || gotMin != minEnc || gotMax != maxEnc {
				t.Fatalf("vec %d len %d (e=%d f=%d): encoded=(exc %d,min %d,max %d) scalar=(exc %d,min %d,max %d)",
					vi, n, e, f, gotExceptions, gotMin, gotMax, exceptions, minEnc, maxEnc)
			}
			refPosition := 0
			for i, value := range vec {
				enc, exc := evalFloat(value, e, f)
				if exc {
					if encodedPos[refPosition] != uint16(i) {
						t.Fatalf("vec %d len %d (e=%d f=%d) exception %d: encoded position=%d scalar=%d",
							vi, n, e, f, refPosition, encodedPos[refPosition], i)
					}
					refPosition++
				} else if encoded[i] != uint32(enc) {
					t.Fatalf("vec %d len %d (e=%d f=%d) lane %d: raw encoding=%d scalar=%d",
						vi, n, e, f, i, encoded[i], uint32(enc))
				}
			}
			if exceptions == n {
				minEnc = 0
			}
			encodeFloatDeltasScalar(vec, e, f, minEnc, refDeltas, refPos)
			encodeFloatDeltas(vec, e, f, minEnc, gotDeltas, gotPos)
			for i := range refPos {
				if gotPos[i] != refPos[i] {
					t.Fatalf("vec %d len %d (e=%d f=%d) exception %d: position AVX2=%d scalar=%d",
						vi, n, e, f, i, gotPos[i], refPos[i])
				}
			}
			for i := range n {
				if gotDeltas[i] != refDeltas[i] {
					t.Fatalf("vec %d len %d (e=%d f=%d) lane %d: delta AVX2=%d scalar=%d",
						vi, n, e, f, i, gotDeltas[i], refDeltas[i])
				}
			}
		}
	}
}

func TestAnalyzeFloatAVX2Boundaries(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	for _, pair := range allValidFloatPairs {
		e, f := pair[0], pair[1]
		values := []float32{
			math.Float32frombits(0x00000001), // smallest positive subnormal
			math.Float32frombits(0x80000001), // smallest negative subnormal
			math.Float32frombits(0x007fffff), // largest positive subnormal
			math.Float32frombits(0x807fffff), // largest negative subnormal
			math.Float32frombits(0x7fc00001), // positive quiet NaN with payload
			math.Float32frombits(0xffc12345), // negative quiet NaN with payload
			math.Float32frombits(0x7f800001), // positive signaling NaN with payload
			math.Float32frombits(0xff800001), // negative signaling NaN with payload
		}

		// Map exact half-integers back through the decode scaling expression,
		// then test the adjacent representable inputs on both sides.
		for _, integer := range []int32{-4194303, -1025, -2, -1, 0, 1, 2, 1024, 4194302} {
			half := float32(integer) + 0.5
			value := half * floatPow10[f] * floatPow10Negative[e]
			values = appendFloatNeighbors(values, value)
		}

		// Exercise the last representable values on each side of both scaled
		// encoding limits, including values immediately outside the limits.
		for _, limit := range []float32{floatEncodingLowerLimit, floatEncodingUpperLimit} {
			for _, scaled := range []float32{
				math.Nextafter32(limit, float32(math.Inf(-1))),
				limit,
				math.Nextafter32(limit, float32(math.Inf(1))),
			} {
				value := scaled * floatPow10[f] * floatPow10Negative[e]
				values = appendFloatNeighbors(values, value)
			}
		}

		for len(values)%8 != 0 {
			values = append(values, 0)
		}
		checkFloatOptimizedPaths(t, values, e, f, true)
	}
}

func TestAnalyzeFloatAVX2AllExceptionsAndTails(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	exceptions := []float32{
		math.Float32frombits(0x7fc00001),
		math.Float32frombits(0xffc12345),
		math.Float32frombits(0x7f800001),
		math.Float32frombits(0xff800001),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		math.Float32frombits(0x80000000),
	}
	for _, n := range []int{8, 9, 15, 16, 17, 23, 1024, 1031} {
		values := make([]float32, n)
		for i := range values {
			values[i] = exceptions[i%len(exceptions)]
		}
		for _, pair := range allValidFloatPairs {
			checkFloatOptimizedPaths(t, values, pair[0], pair[1], n%8 == 0)
		}
	}
}

func appendFloatNeighbors(dst []float32, value float32) []float32 {
	return append(dst,
		math.Nextafter32(value, float32(math.Inf(-1))),
		value,
		math.Nextafter32(value, float32(math.Inf(1))),
	)
}

func checkFloatOptimizedPaths(t *testing.T, values []float32, e, f int, directAVX2 bool) {
	t.Helper()
	exceptions, minEnc, maxEnc := analyzeFloatScalar(values, e, f)
	if directAVX2 {
		gotExceptions, gotMin, gotMax := callAVX2(values, e, f)
		if gotExceptions != exceptions || gotMin != minEnc || gotMax != maxEnc {
			t.Fatalf("len=%d (e=%d f=%d): AVX2=(exc %d,min %d,max %d) scalar=(exc %d,min %d,max %d)",
				len(values), e, f, gotExceptions, gotMin, gotMax, exceptions, minEnc, maxEnc)
		}
	}
	gotExceptions, gotMin, gotMax := analyzeFloat(values, e, f)
	if gotExceptions != exceptions || gotMin != minEnc || gotMax != maxEnc {
		t.Fatalf("len=%d (e=%d f=%d): dispatch=(exc %d,min %d,max %d) scalar=(exc %d,min %d,max %d)",
			len(values), e, f, gotExceptions, gotMin, gotMax, exceptions, minEnc, maxEnc)
	}

	encoded := make([]uint32, len(values))
	encodedPos := make([]uint16, exceptions)
	gotExceptions, gotMin, gotMax = analyzeFloatEncoded(values, e, f, encoded, encodedPos)
	if gotExceptions != exceptions || gotMin != minEnc || gotMax != maxEnc {
		t.Fatalf("len=%d (e=%d f=%d): encoded=(exc %d,min %d,max %d) scalar=(exc %d,min %d,max %d)",
			len(values), e, f, gotExceptions, gotMin, gotMax, exceptions, minEnc, maxEnc)
	}
	position := 0
	for i, value := range values {
		enc, exc := evalFloat(value, e, f)
		if exc {
			if encodedPos[position] != uint16(i) {
				t.Fatalf("len=%d (e=%d f=%d) exception=%d: position=%d want=%d", len(values), e, f, position, encodedPos[position], i)
			}
			position++
		} else if encoded[i] != uint32(enc) {
			t.Fatalf("len=%d (e=%d f=%d) lane=%d: encoded=%d want=%d", len(values), e, f, i, encoded[i], uint32(enc))
		}
	}

	if exceptions == len(values) {
		minEnc = 0
	}
	wantDeltas := make([]uint32, len(values))
	gotDeltas := make([]uint32, len(values))
	for i := range gotDeltas {
		gotDeltas[i] = 0xdeadbeef
	}
	wantPos := make([]uint16, exceptions)
	gotPos := make([]uint16, exceptions)
	encodeFloatDeltasScalar(values, e, f, minEnc, wantDeltas, wantPos)
	encodeFloatDeltas(values, e, f, minEnc, gotDeltas, gotPos)
	for i := range values {
		if gotDeltas[i] != wantDeltas[i] {
			t.Fatalf("len=%d (e=%d f=%d) lane=%d: delta=%d want=%d", len(values), e, f, i, gotDeltas[i], wantDeltas[i])
		}
	}
	for i := range wantPos {
		if gotPos[i] != wantPos[i] {
			t.Fatalf("len=%d (e=%d f=%d) exception=%d: delta position=%d want=%d", len(values), e, f, i, gotPos[i], wantPos[i])
		}
	}
}
