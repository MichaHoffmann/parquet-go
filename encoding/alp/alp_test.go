package alp_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/parquet-go/parquet-go/encoding/alp"
	"github.com/parquet-go/parquet-go/encoding/test"
)

func TestEncodeFloat(t *testing.T) {
	test.EncodeFloat(t, new(alp.Encoding), 0, 100)
}

func TestEncodeDouble(t *testing.T) {
	test.EncodeDouble(t, new(alp.Encoding), 0, 100)
}

// TestDeterministicEncode asserts that encoding the same input twice yields
// byte-identical output. This guards deterministic sampled-preset ranking. It
// covers both a large input that triggers preset building (> 8*1024 values,
// i.e. > 8 vectors) and a small one that stays in the full-search phase.
func TestDeterministicEncode(t *testing.T) {
	// A mix of decimal scales, some exceptions, and negatives so that different
	// vectors prefer different (exponent, factor) presets.
	genDoubles := func(n int) []float64 {
		out := make([]float64, n)
		for i := range out {
			switch i % 7 {
			case 0:
				out[i] = float64(i%1000) * 0.01
			case 1:
				out[i] = float64(i%500) * 0.001
			case 2:
				out[i] = -float64(i%777) * 0.1
			case 3:
				out[i] = math.Pi * float64(i%13)
			case 4:
				out[i] = float64(i%9999) * 1e-6
			case 5:
				out[i] = float64(i) // integral
			default:
				out[i] = float64(i%321) * 0.025
			}
		}
		return out
	}
	genFloats := func(n int) []float32 {
		d := genDoubles(n)
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(d[i])
		}
		return out
	}

	for _, n := range []int{5 * 1024, 130 * 1024} {
		phase := "full-search"
		if n > 8*1024 {
			phase = "preset"
		}

		dbl := genDoubles(n)
		var e alp.Encoding
		a, err := e.EncodeDouble(nil, dbl)
		if err != nil {
			t.Fatalf("EncodeDouble(%d, %s): %v", n, phase, err)
		}
		b, err := e.EncodeDouble(nil, dbl)
		if err != nil {
			t.Fatalf("EncodeDouble(%d, %s): %v", n, phase, err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("EncodeDouble not deterministic (n=%d, %s): %d vs %d bytes differ", n, phase, len(a), len(b))
		}

		flt := genFloats(n)
		af, err := e.EncodeFloat(nil, flt)
		if err != nil {
			t.Fatalf("EncodeFloat(%d, %s): %v", n, phase, err)
		}
		bf, err := e.EncodeFloat(nil, flt)
		if err != nil {
			t.Fatalf("EncodeFloat(%d, %s): %v", n, phase, err)
		}
		if !bytes.Equal(af, bf) {
			t.Fatalf("EncodeFloat not deterministic (n=%d, %s): %d vs %d bytes differ", n, phase, len(af), len(bf))
		}

		// Round-trip must stay lossless through the new encode path.
		gotD, err := e.DecodeDouble(nil, a)
		if err != nil {
			t.Fatalf("DecodeDouble(%d, %s): %v", n, phase, err)
		}
		for i := range dbl {
			if math.Float64bits(gotD[i]) != math.Float64bits(dbl[i]) {
				t.Fatalf("DecodeDouble lossy at %d (n=%d, %s): got %v want %v", i, n, phase, gotD[i], dbl[i])
			}
		}
		gotF, err := e.DecodeFloat(nil, af)
		if err != nil {
			t.Fatalf("DecodeFloat(%d, %s): %v", n, phase, err)
		}
		for i := range flt {
			if math.Float32bits(gotF[i]) != math.Float32bits(flt[i]) {
				t.Fatalf("DecodeFloat lossy at %d (n=%d, %s): got %v want %v", i, n, phase, gotF[i], flt[i])
			}
		}
	}
}

func TestStatefulEncoderDerivesPresetsPerPage(t *testing.T) {
	const sampleValues = 9 * 1024 // more than the eight-vector sampling threshold

	doubles := func(n int, scale float64) []float64 {
		values := make([]float64, n)
		for i := range values {
			values[i] = float64(i%1000) * scale
		}
		return values
	}
	floats := func(n int, scale float32) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(i%1000) * scale
		}
		return values
	}

	var e alp.Encoding
	t.Run("double", func(t *testing.T) {
		state := e.NewStatefulEncoder()
		if _, err := state.EncodeDouble(nil, doubles(sampleValues, 1)); err != nil {
			t.Fatal(err)
		}
		decimal := doubles(sampleValues, 0.001)
		got, err := state.EncodeDouble(nil, decimal)
		if err != nil {
			t.Fatal(err)
		}
		want, err := e.EncodeDouble(nil, decimal)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatal("second double page did not derive page-local presets")
		}
	})

	t.Run("float", func(t *testing.T) {
		state := e.NewStatefulEncoder()
		if _, err := state.EncodeFloat(nil, floats(sampleValues, 1)); err != nil {
			t.Fatal(err)
		}
		decimal := floats(sampleValues, 0.001)
		got, err := state.EncodeFloat(nil, decimal)
		if err != nil {
			t.Fatal(err)
		}
		want, err := e.EncodeFloat(nil, decimal)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatal("second float page did not derive page-local presets")
		}
	})
}
