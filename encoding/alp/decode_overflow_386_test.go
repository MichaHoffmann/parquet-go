package alp_test

import (
	"encoding/binary"
	"testing"

	"github.com/parquet-go/parquet-go/encoding/alp"
)

func constantVectorPage386(elementCount, vectorBytes int) []byte {
	const (
		logVectorSize = 15
		vectorSize    = 1 << logVectorSize
	)
	numVectors := (elementCount + vectorSize - 1) / vectorSize
	offsetTableSize := numVectors * 4
	page := make([]byte, 7+offsetTableSize, 7+offsetTableSize+numVectors*vectorBytes)
	page[2] = logVectorSize
	binary.LittleEndian.PutUint32(page[3:], uint32(elementCount))
	for i := range numVectors {
		binary.LittleEndian.PutUint32(page[7+i*4:], uint32(len(page)-7))
		page = append(page, make([]byte, vectorBytes)...)
	}
	return page
}

func TestDecodeFloatOutputSizeOverflow(t *testing.T) {
	page := constantVectorPage386(1<<30, 9)
	var e alp.Encoding
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("decode panicked: %v", recovered)
		}
	}()

	got, err := e.DecodeFloat([]float32{1}, page)
	if err == nil {
		t.Fatal("expected error")
	}
	if len(got) != 0 {
		t.Fatalf("output length: got %d want 0", len(got))
	}
}

func TestDecodeDoubleOutputSizeOverflow(t *testing.T) {
	page := constantVectorPage386(1<<29, 13)
	var e alp.Encoding
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("decode panicked: %v", recovered)
		}
	}()

	got, err := e.DecodeDouble([]float64{1}, page)
	if err == nil {
		t.Fatal("expected error")
	}
	if len(got) != 0 {
		t.Fatalf("output length: got %d want 0", len(got))
	}
}
