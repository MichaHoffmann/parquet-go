//go:build !purego

package alp

import (
	"math"
	"math/rand"
	"testing"
)

func TestAnalyzeFloatBoundedAVX2CompleteDifferential(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	specials := []float32{
		0, math.Float32frombits(floatNegativeZeroBits), 1, -1, 0.1, -0.1,
		math.Float32frombits(0x7fc00001), math.Float32frombits(0xff800001),
		float32(math.Inf(1)), float32(math.Inf(-1)),
		floatEncodingUpperLimit, floatEncodingLowerLimit,
		math.Nextafter32(floatEncodingUpperLimit, float32(math.Inf(1))),
		math.Nextafter32(floatEncodingLowerLimit, float32(math.Inf(-1))),
	}
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{8, 9, 15, 16, 63, 64, 127, 128, 129, 255, 256, 257, 1023, 1024, 1031} {
		values := make([]float32, n)
		for i := range values {
			if i%5 == 0 {
				values[i] = specials[(i/5)%len(specials)]
			} else {
				values[i] = float32(rng.Intn(200000)-100000) * 0.001
			}
		}
		for _, pair := range allValidFloatPairs {
			wantExceptions, wantMin, wantMax := analyzeFloatScalar(values, pair[0], pair[1])
			exceptions, minEnc, maxEnc, complete, processed := analyzeFloatBounded(values, pair[0], pair[1], math.MaxInt64)
			if !complete || processed != n || exceptions != wantExceptions || minEnc != wantMin || maxEnc != wantMax {
				t.Fatalf("len=%d e=%d f=%d: got (%d,%d,%d complete=%t processed=%d), want (%d,%d,%d complete=true processed=%d)",
					n, pair[0], pair[1], exceptions, minEnc, maxEnc, complete, processed,
					wantExceptions, wantMin, wantMax, n)
			}
		}
	}
}

func TestAnalyzeFloatBoundedAVX2CutoffExactness(t *testing.T) {
	if !useAnalyzeAVX2 {
		t.Skip("AVX2 and POPCNT not available")
	}

	lengths := []int{63, 64, 65, 127, 128, 129, 135, 255, 256, 257, 263}
	for _, n := range lengths {
		values := makeFloatTestValues(n)
		values[n/2] = math.Float32frombits(0x7fc00001)
		for _, pair := range allValidFloatPairs {
			wantExceptions, wantMin, wantMax := analyzeFloatScalar(values, pair[0], pair[1])
			score := floatAnalysisLowerBound(n, wantExceptions, wantMin, wantMax)
			for _, cutoff := range []int64{score - 1, score, score + 1} {
				exceptions, minEnc, maxEnc, complete, processed := analyzeFloatBounded(values, pair[0], pair[1], cutoff)
				if cutoff >= score {
					if !complete || processed != n || exceptions != wantExceptions || minEnc != wantMin || maxEnc != wantMax {
						t.Fatalf("len=%d e=%d f=%d cutoff=%d: inexact completed result", n, pair[0], pair[1], cutoff)
					}
					continue
				}
				if complete || processed <= 0 || processed > n {
					t.Fatalf("len=%d e=%d f=%d cutoff=%d: complete=%t processed=%d", n, pair[0], pair[1], cutoff, complete, processed)
				}
				n8 := n &^ 7
				if processed < n8 && processed%128 != 0 {
					t.Fatalf("len=%d e=%d f=%d cutoff=%d: processed=%d is not a checkpoint", n, pair[0], pair[1], cutoff, processed)
				}
				if floatAnalysisLowerBound(n, exceptions, minEnc, maxEnc) <= cutoff {
					t.Fatalf("len=%d e=%d f=%d cutoff=%d: pruned partial result does not prove loss", n, pair[0], pair[1], cutoff)
				}
				if score <= cutoff {
					t.Fatalf("len=%d e=%d f=%d cutoff=%d: pruned candidate could win with score=%d", n, pair[0], pair[1], cutoff, score)
				}
			}
		}
	}
}

func TestAnalyzeFloatBoundedAVX2AllExceptionsAndFallback(t *testing.T) {
	exceptions := []float32{
		math.Float32frombits(0x7fc00001), float32(math.Inf(1)), float32(math.Inf(-1)),
		math.Float32frombits(floatNegativeZeroBits), math.Float32frombits(0x7f800001),
	}
	values := make([]float32, 263)
	for i := range values {
		values[i] = exceptions[i%len(exceptions)]
	}

	old := useAnalyzeAVX2
	defer func() { useAnalyzeAVX2 = old }()
	for _, enabled := range []bool{false, old} {
		useAnalyzeAVX2 = enabled
		wantExceptions, wantMin, wantMax := analyzeFloatScalar(values, 0, 0)
		exceptions, minEnc, maxEnc, complete, processed := analyzeFloatBounded(values, 0, 0, math.MaxInt64)
		if !complete || processed != len(values) || exceptions != wantExceptions || minEnc != wantMin || maxEnc != wantMax {
			t.Fatalf("enabled=%t: got (%d,%d,%d complete=%t processed=%d), want (%d,%d,%d)",
				enabled, exceptions, minEnc, maxEnc, complete, processed, wantExceptions, wantMin, wantMax)
		}
	}
}
