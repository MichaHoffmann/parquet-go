package alp

import (
	"math"
	"testing"
)

func TestAnalyzeFloatBoundedCutoffs(t *testing.T) {
	tests := []struct {
		name   string
		values []float32
		e, f   int
	}{
		{"mixed", []float32{1.25, -2.5, 3.75, float32(math.Pi), 0, 1000.125}, 2, 0},
		{"constant", []float32{12.5, 12.5, 12.5, 12.5}, 1, 0},
		{"all-exception", []float32{
			math.Float32frombits(0x7fc00001),
			float32(math.Inf(1)),
			float32(math.Inf(-1)),
			math.Float32frombits(floatNegativeZeroBits),
		}, 0, 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wantExceptions, wantMin, wantMax := analyzeFloatScalar(test.values, test.e, test.f)
			score := floatAnalysisLowerBound(len(test.values), wantExceptions, wantMin, wantMax)
			for _, cutoff := range []int64{score - 1, score, score + 1} {
				exceptions, minEnc, maxEnc, complete, processed := analyzeFloatBounded(test.values, test.e, test.f, cutoff)
				if cutoff < score {
					if complete || processed > len(test.values) {
						t.Fatalf("cutoff %d: complete=%t processed=%d, want pruned", cutoff, complete, processed)
					}
					continue
				}
				if !complete || processed != len(test.values) {
					t.Fatalf("cutoff %d: complete=%t processed=%d, want complete", cutoff, complete, processed)
				}
				if exceptions != wantExceptions || minEnc != wantMin || maxEnc != wantMax {
					t.Fatalf("cutoff %d: got (%d,%d,%d), want (%d,%d,%d)", cutoff, exceptions, minEnc, maxEnc, wantExceptions, wantMin, wantMax)
				}
			}
		})
	}
}

func TestFloatAnalysisLowerBoundMonotone(t *testing.T) {
	values := []float32{1.25, float32(math.Pi), -8.5, float32(math.Inf(1)), 100.125, 0.5}
	const e, f = 2, 0
	minEnc, maxEnc := int32(math.MaxInt32), int32(math.MinInt32)
	exceptions := 0
	previous := int64(-1)
	for i, value := range values {
		enc, exc := evalFloat(value, e, f)
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
		bound := floatAnalysisLowerBound(len(values), exceptions, minEnc, maxEnc)
		if bound < previous {
			t.Fatalf("lane %d: lower bound decreased from %d to %d", i, previous, bound)
		}
		previous = bound
	}
	wantExceptions, wantMin, wantMax := analyzeFloatScalar(values, e, f)
	want := floatAnalysisLowerBound(len(values), wantExceptions, wantMin, wantMax)
	if previous != want {
		t.Fatalf("final lower bound=%d, want score=%d", previous, want)
	}
}

func TestAnalyzeDoubleBoundedCutoffs(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		e, f   int
	}{
		{"mixed", []float64{1.25, -2.5, 3.75, math.Pi, 0, 1000.125}, 2, 0},
		{"constant", []float64{12.5, 12.5, 12.5, 12.5}, 1, 0},
		{"all-exception", specialExceptionDoubles(4), 0, 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wantExceptions, wantMin, wantMax := analyzeDoubleScalar(test.values, test.e, test.f)
			score := doubleAnalysisLowerBound(len(test.values), wantExceptions, wantMin, wantMax)
			for _, cutoff := range []int64{score - 1, score, score + 1} {
				exceptions, minEnc, maxEnc, complete, processed := analyzeDoubleBounded(test.values, test.e, test.f, cutoff)
				if cutoff < score {
					if complete || processed > len(test.values) {
						t.Fatalf("cutoff %d: complete=%t processed=%d, want pruned", cutoff, complete, processed)
					}
					continue
				}
				if !complete || processed != len(test.values) {
					t.Fatalf("cutoff %d: complete=%t processed=%d, want complete", cutoff, complete, processed)
				}
				if exceptions != wantExceptions || minEnc != wantMin || maxEnc != wantMax {
					t.Fatalf("cutoff %d: got (%d,%d,%d), want (%d,%d,%d)", cutoff, exceptions, minEnc, maxEnc, wantExceptions, wantMin, wantMax)
				}
			}
		})
	}
}

func makeFloatTestValues(n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		switch i % 5 {
		case 0:
			values[i] = float32(i%1000) * 0.01
		case 1:
			values[i] = -float32(i%700) * 0.1
		case 2:
			values[i] = float32(math.Pi) * float32(i%17)
		case 3:
			values[i] = float32(i % 100)
		default:
			values[i] = float32(i%300) * 0.001
		}
	}
	return values
}
