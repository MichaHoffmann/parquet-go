package alp_test

import (
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"testing"

	"github.com/parquet-go/parquet-go/encoding"
	"github.com/parquet-go/parquet-go/encoding/alp"
)

func validDoublePage(t *testing.T) []byte {
	t.Helper()
	var e alp.Encoding
	buf, err := e.EncodeDouble(nil, []float64{1.5, 2.5, 3.5, 4.5, math.NaN()})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return buf
}

func constantDoublePage() []byte {
	page := []byte{
		0, 0, 3, 16, 0, 0, 0,
		8, 0, 0, 0, 21, 0, 0, 0,
	}
	for _, value := range []uint64{5, 7} {
		page = append(page, 0, 0, 0, 0)
		page = binary.LittleEndian.AppendUint64(page, value)
		page = append(page, 0)
	}
	return page
}

func TestDecodeInvalid(t *testing.T) {
	var e alp.Encoding
	assertErr := func(t *testing.T, page []byte) {
		t.Helper()
		defer func() {
			if recovered := recover(); recovered != nil {
				t.Fatalf("decode panicked: %v", recovered)
			}
		}()
		if _, err := e.DecodeDouble(nil, page); err == nil {
			t.Fatal("expected error")
		}
	}

	t.Run("truncated header", func(t *testing.T) {
		assertErr(t, []byte{0x00, 0x00, 0x0a})
	})
	t.Run("compression mode", func(t *testing.T) {
		page := validDoublePage(t)
		page[0] = 0x01
		assertErr(t, page)
	})
	t.Run("integer encoding", func(t *testing.T) {
		page := validDoublePage(t)
		page[1] = 0x01
		assertErr(t, page)
	})
	t.Run("vector size too small", func(t *testing.T) {
		page := validDoublePage(t)
		page[2] = 0x02
		assertErr(t, page)
	})
	t.Run("vector size too large", func(t *testing.T) {
		page := validDoublePage(t)
		page[2] = 0x10
		assertErr(t, page)
	})
	t.Run("negative element count", func(t *testing.T) {
		page := validDoublePage(t)
		page[3], page[4], page[5], page[6] = 0xff, 0xff, 0xff, 0xff
		assertErr(t, page)
	})
	t.Run("exponent too large", func(t *testing.T) {
		page := validDoublePage(t)
		page[11] = 99
		assertErr(t, page)
	})
	t.Run("factor greater than exponent", func(t *testing.T) {
		page := validDoublePage(t)
		page[11] = 1
		page[12] = 5
		assertErr(t, page)
	})
	t.Run("too many exceptions", func(t *testing.T) {
		page := validDoublePage(t)
		page[13], page[14] = 0xff, 0xff
		assertErr(t, page)
	})
	t.Run("exception position out of bounds", func(t *testing.T) {
		page := []byte{
			0x00, 0x00, 0x0a,
			0x02, 0x00, 0x00, 0x00,
			0x04, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x01, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00,
			0x63, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		}
		assertErr(t, page)
	})
	t.Run("truncated offset array", func(t *testing.T) {
		assertErr(t, []byte{
			0x00, 0x00, 0x0a,
			0x05, 0x00, 0x00, 0x00,
			0x04, 0x00,
		})
	})
	t.Run("wraps invalid argument", func(t *testing.T) {
		_, err := e.DecodeDouble(nil, []byte{0x00})
		if !errors.Is(err, encoding.ErrInvalidArgument) {
			t.Fatalf("expected ErrInvalidArgument, got %v", err)
		}
	})
}

func TestDecodeInvalidVectorOffsets(t *testing.T) {
	valid := constantDoublePage()
	tests := []struct {
		name    string
		corrupt func([]byte)
	}{
		{name: "aliased", corrupt: func(page []byte) { binary.LittleEndian.PutUint32(page[11:], 8) }},
		{name: "reversed", corrupt: func(page []byte) { binary.LittleEndian.PutUint32(page[11:], 4) }},
		{name: "first after body start", corrupt: func(page []byte) { binary.LittleEndian.PutUint32(page[7:], 9) }},
		{name: "first region truncated", corrupt: func(page []byte) { binary.LittleEndian.PutUint32(page[11:], 12) }},
		{name: "offset past body", corrupt: func(page []byte) { binary.LittleEndian.PutUint32(page[11:], uint32(len(page))) }},
	}

	var e alp.Encoding
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			page := append([]byte(nil), valid...)
			test.corrupt(page)
			if _, err := e.DecodeDouble(nil, page); err == nil {
				t.Fatal("malformed offsets were accepted")
			}
		})
	}
}

func TestDecodeInvalidExceptionRegionCannotAliasNextVector(t *testing.T) {
	page := []byte{
		0, 0, 3, 16, 0, 0, 0,
		8, 0, 0, 0, 17, 0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
	}
	var e alp.Encoding
	if _, err := e.DecodeFloat(nil, page); err == nil {
		t.Fatal("exception data crossing into the next vector was accepted")
	}
}

func TestDecodeInvalidHugeAllocation(t *testing.T) {
	var e alp.Encoding
	noPanicErr := func(t *testing.T, fn func() error) {
		t.Helper()
		defer func() {
			if recovered := recover(); recovered != nil {
				t.Fatalf("decode panicked: %v", recovered)
			}
		}()
		if err := fn(); err == nil {
			t.Fatal("expected error")
		}
	}

	const logVectorSize = 15
	const vectorSize = 1 << logVectorSize
	const numElements = 2_000_000_000
	t.Run("empty body", func(t *testing.T) {
		page := make([]byte, 7)
		page[2] = logVectorSize
		binary.LittleEndian.PutUint32(page[3:], numElements)
		t.Run("double", func(t *testing.T) {
			noPanicErr(t, func() error { _, err := e.DecodeDouble(nil, page); return err })
		})
		t.Run("float", func(t *testing.T) {
			noPanicErr(t, func() error { _, err := e.DecodeFloat(nil, page); return err })
		})
	})
	t.Run("lying body length", func(t *testing.T) {
		numVectors := (numElements + vectorSize - 1) / vectorSize
		page := make([]byte, 7+numVectors*4+8)
		page[2] = logVectorSize
		binary.LittleEndian.PutUint32(page[3:], numElements)
		t.Run("double", func(t *testing.T) {
			noPanicErr(t, func() error { _, err := e.DecodeDouble(nil, page); return err })
		})
		t.Run("float", func(t *testing.T) {
			noPanicErr(t, func() error { _, err := e.DecodeFloat(nil, page); return err })
		})
	})
}

func TestDecodeLateMalformedVectorBeforeOutputAllocation(t *testing.T) {
	const logVectorSize = 15
	const vectorSize = 1 << logVectorSize
	const numElements = math.MaxInt32
	const headerSize = 7
	const alpInfoSize = 4
	const floatForInfoSize = 5
	const doubleForInfoSize = 9
	numVectors := (numElements + vectorSize - 1) / vectorSize

	makePage := func(forInfoSize, invalidBitWidth int) []byte {
		vectorBytes := alpInfoSize + forInfoSize
		offsetTableSize := numVectors * 4
		page := make([]byte, headerSize+offsetTableSize+numVectors*vectorBytes)
		page[2] = logVectorSize
		binary.LittleEndian.PutUint32(page[3:], numElements)
		for v := range numVectors {
			binary.LittleEndian.PutUint32(page[headerSize+v*4:], uint32(offsetTableSize+v*vectorBytes))
		}
		lastVector := headerSize + offsetTableSize + (numVectors-1)*vectorBytes
		page[lastVector+alpInfoSize+forInfoSize-1] = byte(invalidBitWidth)
		return page
	}

	floatPage := makePage(floatForInfoSize, 33)
	doublePage := makePage(doubleForInfoSize, 65)
	var e alp.Encoding
	for _, test := range []struct {
		name   string
		decode func() error
	}{
		{
			name: "float",
			decode: func() error {
				_, err := e.DecodeFloat([]float32{42}, floatPage)
				return err
			},
		},
		{
			name: "double",
			decode: func() error {
				_, err := e.DecodeDouble([]float64{42}, doublePage)
				return err
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("decode panicked: %v", recovered)
				}
			}()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			err := test.decode()
			runtime.ReadMemStats(&after)
			if err == nil {
				t.Fatal("expected error from malformed final vector")
			}
			if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 1<<20 {
				t.Fatalf("decode allocated %d bytes before rejecting malformed final vector", allocated)
			}
		})
	}
}

func FuzzDecodeMalformedOffsets(f *testing.F) {
	valid := constantDoublePage()
	f.Add(valid)
	aliased := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(aliased[11:], 8)
	f.Add(aliased)
	reversed := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(reversed[11:], 4)
	f.Add(reversed)

	f.Fuzz(func(t *testing.T, page []byte) {
		var e alp.Encoding
		_, _ = e.DecodeDouble(nil, page)
		_, _ = e.DecodeFloat(nil, page)
	})
}
