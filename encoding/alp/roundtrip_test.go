package alp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/parquet-go/parquet-go/encoding/alp"
)

// roundTripDouble encodes then decodes src and asserts bit-exact equality.
func roundTripDouble(t *testing.T, src []float64) {
	t.Helper()
	var e alp.Encoding
	buf, err := e.EncodeDouble(nil, src)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := e.DecodeDouble(nil, buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(src) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(src))
	}
	for i := range src {
		if math.Float64bits(got[i]) != math.Float64bits(src[i]) {
			t.Fatalf("value %d: got bits %#016x (%v) want %#016x (%v)",
				i, math.Float64bits(got[i]), got[i], math.Float64bits(src[i]), src[i])
		}
	}
}

// roundTripFloat encodes then decodes src and asserts bit-exact equality.
func roundTripFloat(t *testing.T, src []float32) {
	t.Helper()
	var e alp.Encoding
	buf, err := e.EncodeFloat(nil, src)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := e.DecodeFloat(nil, buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(src) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(src))
	}
	for i := range src {
		if math.Float32bits(got[i]) != math.Float32bits(src[i]) {
			t.Fatalf("value %d: got bits %#08x (%v) want %#08x (%v)",
				i, math.Float32bits(got[i]), got[i], math.Float32bits(src[i]), src[i])
		}
	}
}

func TestDoubleSpecialValues(t *testing.T) {
	nonCanonicalNaN := math.Float64frombits(0x7FF8_0000_DEAD_BEEF)
	tests := []struct {
		name string
		src  []float64
	}{
		{name: "empty", src: []float64{}},
		{name: "all NaN", src: []float64{math.NaN(), math.NaN(), math.NaN()}},
		{name: "non-canonical NaN", src: []float64{nonCanonicalNaN, 1.5, nonCanonicalNaN}},
		{name: "constant", src: []float64{3.14, 3.14, 3.14, 3.14, 3.14}},
		{name: "positive infinity", src: []float64{math.Inf(1), 1, 2}},
		{name: "negative infinity", src: []float64{math.Inf(-1), 1, 2}},
		{name: "negative zero", src: []float64{math.Copysign(0, -1), 0, 1}},
		{name: "positive zero", src: []float64{0, 0, 0}},
		{name: "maximum double", src: []float64{math.MaxFloat64, 1, 2, 3}},
		{name: "smallest", src: []float64{math.SmallestNonzeroFloat64, 1, 2}},
		{name: "mixed signs", src: []float64{-1.5, 2.5, -3.5, 4.5, -5.5}},
		{name: "all exceptions", src: []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1)}},
		{name: "decimals", src: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 1.25, 12.75, 100.125}},
		{name: "large integers", src: []float64{1e15, 2e15, 3e15}},
		{name: "tiny fractions", src: []float64{1e-15, 2e-15, 3e-15}},
		{name: "embedded negative zero", src: []float64{1.0, math.Copysign(0, -1), 2.0, 0.0, 3.0}},
		{name: "single", src: []float64{42.0}},
		{name: "single NaN", src: []float64{math.NaN()}},
		{name: "single negative zero", src: []float64{math.Copysign(0, -1)}},
		{name: "overflow bounds", src: []float64{9223372036854774784.0, -9223372036854774784.0, 1e300, -1e300}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) { roundTripDouble(t, test.src) })
	}
}

func TestFloatSpecialValues(t *testing.T) {
	nonCanonicalNaN := math.Float32frombits(0x7FC0_DEAD)
	tests := []struct {
		name string
		src  []float32
	}{
		{name: "empty", src: []float32{}},
		{name: "all NaN", src: []float32{float32(math.NaN()), float32(math.NaN())}},
		{name: "non-canonical NaN", src: []float32{nonCanonicalNaN, 1.5, nonCanonicalNaN}},
		{name: "constant", src: []float32{3.14, 3.14, 3.14, 3.14}},
		{name: "positive infinity", src: []float32{float32(math.Inf(1)), 1, 2}},
		{name: "negative infinity", src: []float32{float32(math.Inf(-1)), 1, 2}},
		{name: "negative zero", src: []float32{float32(math.Copysign(0, -1)), 0, 1}},
		{name: "positive zero", src: []float32{0, 0, 0}},
		{name: "maximum float", src: []float32{math.MaxFloat32, 1, 2, 3}},
		{name: "smallest", src: []float32{math.SmallestNonzeroFloat32, 1, 2}},
		{name: "mixed signs", src: []float32{-1.5, 2.5, -3.5, 4.5, -5.5}},
		{name: "all exceptions", src: []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Copysign(0, -1))}},
		{name: "decimals", src: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 1.25, 12.75, 100.125}},
		{name: "single", src: []float32{42.0}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) { roundTripFloat(t, test.src) })
	}
}

// TestPartialVectorRemainders exercises every partial-vector remainder around
// the 1024-value vector boundary.
func TestPartialVectorRemainders(t *testing.T) {
	lengths := []int{}
	for n := 0; n <= 20; n++ {
		lengths = append(lengths, n)
	}
	for _, base := range []int{1024, 2048, 3072} {
		for r := -2; r <= 3; r++ {
			if base+r >= 0 {
				lengths = append(lengths, base+r)
			}
		}
	}
	for _, n := range lengths {
		t.Run(fmt.Sprintf("double/N=%d", n), func(t *testing.T) {
			src := make([]float64, n)
			for i := range src {
				src[i] = float64(i) * 0.25
			}
			roundTripDouble(t, src)
		})
		t.Run(fmt.Sprintf("float/N=%d", n), func(t *testing.T) {
			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i) * 0.25
			}
			roundTripFloat(t, src)
		})
	}
}

// TestMultiVectorDifferingExponents builds several vectors, each with a
// different natural exponent/magnitude, plus scattered exceptions.
func TestMultiVectorDifferingExponents(t *testing.T) {
	const n = 2600
	src := make([]float64, n)
	for i := range src {
		switch v := i / 1024; v {
		case 0:
			src[i] = float64(i) // integers, exponent 0
		case 1:
			src[i] = float64(i) * 0.001 // 3 decimals
		default:
			src[i] = float64(i) * 12.5 // 1 decimal, larger magnitude
		}
	}
	// Scatter exceptions across vectors.
	src[10] = math.NaN()
	src[1030] = math.Inf(1)
	src[2050] = math.Copysign(0, -1)
	roundTripDouble(t, src)

	fsrc := make([]float32, n)
	for i := range fsrc {
		fsrc[i] = float32(src[i])
	}
	fsrc[10] = float32(math.NaN())
	roundTripFloat(t, fsrc)
}
