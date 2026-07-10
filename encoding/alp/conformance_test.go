package alp

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/parquet-go/bitpack"
)

func TestWorkedExampleGolden(t *testing.T) {
	const nanBits = uint64(0x7ff80000deadbeef)
	values := []float64{1500, math.Float64frombits(nanBits), 2500, 333.3}

	var e Encoding
	got, err := encodeDoublePageWithPresets(&e, nil, values, [][2]int{{4, 3}})
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{
		0x00, 0x00, 0x0a, 0x04, 0x00, 0x00, 0x00,
		0x04, 0x00, 0x00, 0x00,
		0x04, 0x03, 0x01, 0x00,
		0x05, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f,
		0x93, 0xad, 0xc9, 0xd6, 0x28, 0x15, 0x00, 0x00,
		0x01, 0x00,
		0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0xf8, 0x7f,
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("worked example bytes:\n got %x\nwant %x", got, want)
	}
}

func TestSignedIntegerMinimaAreEncodable(t *testing.T) {
	if encoded, exception := evalFloat(float32(math.MinInt32), 0, 0); exception || encoded != math.MinInt32 {
		t.Fatalf("float minimum: encoded=%d exception=%t", encoded, exception)
	}
	if encoded, exception := evalDouble(float64(math.MinInt64), 0, 0); exception || encoded != math.MinInt64 {
		t.Fatalf("double minimum: encoded=%d exception=%t", encoded, exception)
	}
}

func TestExceptionPlaceholders(t *testing.T) {
	t.Run("float bytes", func(t *testing.T) {
		values := []float32{math.Float32frombits(0x7fc01234), 12, float32(math.Inf(1)), 10}
		want := []byte{
			0x00, 0x00, 0x0a, 0x04, 0x00, 0x00, 0x00,
			0x04, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x02, 0x00,
			0x0a, 0x00, 0x00, 0x00, 0x02,
			0x2a,
			0x00, 0x00, 0x02, 0x00,
			0x34, 0x12, 0xc0, 0x7f,
			0x00, 0x00, 0x80, 0x7f,
		}
		tests := []struct {
			name    string
			presets [][2]int
		}{
			{name: "single preset", presets: [][2]int{{0, 0}}},
			{name: "multiple presets", presets: [][2]int{{0, 0}, {0, 0}}},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				var e Encoding
				got, err := encodeFloatPageWithPresets(&e, nil, values, test.presets)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("float placeholder bytes:\n got %x\nwant %x", got, want)
				}
			})
		}
	})

	t.Run("first non-exception", func(t *testing.T) {
		values := []float64{math.NaN(), 12, math.Inf(1), 15}
		var e Encoding
		page, err := encodeDoublePageWithPresets(&e, nil, values, [][2]int{{0, 0}})
		if err != nil {
			t.Fatal(err)
		}
		// FOR=12, width=2, so both exception lanes must contain delta zero.
		packed := page[alpHeaderSize+4+alpInfoSize+doubleForInfoSize:]
		var deltas [4]int64
		bitpack.Unpack(deltas[:], append(append([]byte(nil), packed[0]), make([]byte, bitpack.PaddingInt64)...), 2)
		if deltas != [4]int64{0, 0, 0, 3} {
			t.Fatalf("packed deltas = %v", deltas)
		}
	})

	t.Run("all exceptions", func(t *testing.T) {
		values := []float64{math.NaN(), math.Inf(1), math.Copysign(0, -1)}
		var e Encoding
		page, err := encodeDoublePageWithPresets(&e, nil, values, [][2]int{{4, 3}})
		if err != nil {
			t.Fatal(err)
		}
		vector := page[alpHeaderSize+4:]
		if frame := int64(binary.LittleEndian.Uint64(vector[alpInfoSize:])); frame != 0 {
			t.Fatalf("frame of reference = %d, want 0", frame)
		}
		if width := vector[alpInfoSize+8]; width != 0 {
			t.Fatalf("bit width = %d, want 0", width)
		}
	})
}

func TestDecodeRejectsTrailingBytes(t *testing.T) {
	var e Encoding
	floatPage, err := e.EncodeFloat(nil, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	doublePage, err := e.EncodeDouble(nil, []float64{1, math.NaN(), 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.DecodeFloat(nil, append(floatPage, 0)); err == nil {
		t.Fatal("float page with trailing byte was accepted")
	}
	if _, err := e.DecodeDouble(nil, append(doublePage, 0)); err == nil {
		t.Fatal("double page with trailing byte after exceptions was accepted")
	}
}

func TestZeroElementPayloadIsExactlyHeader(t *testing.T) {
	page := []byte{0, 0, defaultVectorSizeLog, 0, 0, 0, 0}
	var e Encoding
	if _, err := e.DecodeFloat(nil, page); err != nil {
		t.Fatalf("exact header: %v", err)
	}
	if _, err := e.DecodeFloat(nil, append(page, 0)); err == nil {
		t.Fatal("zero-element page with trailing byte was accepted")
	}
}

func TestEncodeOverwritesReusedDestination(t *testing.T) {
	var e Encoding
	floatValues := []float32{1, 2, 3, math.Float32frombits(0x7fc01234)}
	doubleValues := []float64{1500, math.Float64frombits(0x7ff80000deadbeef), 2500, 333.3}

	check := func(t *testing.T, fresh []byte, encode func([]byte) ([]byte, error)) {
		t.Helper()
		dirty := make([]byte, len(fresh), 2*len(fresh))
		for i := range dirty {
			dirty[i] = 0xff
		}
		got, err := encode(dirty)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, fresh) {
			t.Fatalf("reused destination changed encoding:\n got %x\nwant %x", got, fresh)
		}
	}

	freshFloat, err := e.EncodeFloat(nil, floatValues)
	if err != nil {
		t.Fatal(err)
	}
	check(t, freshFloat, func(dst []byte) ([]byte, error) { return e.EncodeFloat(dst, floatValues) })

	freshDouble, err := e.EncodeDouble(nil, doubleValues)
	if err != nil {
		t.Fatal(err)
	}
	check(t, freshDouble, func(dst []byte) ([]byte, error) { return e.EncodeDouble(dst, doubleValues) })
}
