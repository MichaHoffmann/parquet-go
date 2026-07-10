package parquet

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/parquet-go/parquet-go/encoding"
	"github.com/parquet-go/parquet-go/format"
)

func TestALPPageHeaderCountMismatch(t *testing.T) {
	data, err := ALP.EncodeDouble(nil, []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	column := &Column{typ: DoubleType}

	t.Run("v1", func(t *testing.T) {
		header := DataPageHeaderV1{header: &format.DataPageHeader{
			NumValues: 3,
			Encoding:  format.ALP,
		}}
		if _, err := column.DecodeDataPageV1(header, data, nil); !errors.Is(err, ErrCorrupted) {
			t.Fatalf("got %v, want ErrCorrupted", err)
		}
	})

	t.Run("v2", func(t *testing.T) {
		header := DataPageHeaderV2{header: &format.DataPageHeaderV2{
			NumValues: 5,
			NumRows:   5,
			Encoding:  format.ALP,
		}}
		if _, err := column.DecodeDataPageV2(header, data, nil); !errors.Is(err, ErrCorrupted) {
			t.Fatalf("got %v, want ErrCorrupted", err)
		}
	})
}

func TestALPPageHeaderCountPreflight(t *testing.T) {
	data, err := ALP.EncodeDouble(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	column := &Column{typ: DoubleType}

	tests := []struct {
		name   string
		decode func() (Page, error)
	}{
		{"v1", func() (Page, error) {
			return column.DecodeDataPageV1(DataPageHeaderV1{header: &format.DataPageHeader{
				NumValues: math.MaxInt32,
				Encoding:  format.ALP,
			}}, data, nil)
		}},
		{"v2", func() (Page, error) {
			return column.DecodeDataPageV2(DataPageHeaderV2{header: &format.DataPageHeaderV2{
				NumValues: math.MaxInt32,
				NumRows:   math.MaxInt32,
				Encoding:  format.ALP,
			}}, data, nil)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := test.decode(); !errors.Is(err, ErrCorrupted) {
				t.Fatalf("got %v, want ErrCorrupted", err)
			}
		})
	}
}

func TestALPPageEqualHugeCountsValidatePayloadBeforeAllocation(t *testing.T) {
	data := make([]byte, 7)
	data[2] = 10
	binary.LittleEndian.PutUint32(data[3:], math.MaxInt32)
	column := &Column{typ: DoubleType}

	tests := []struct {
		name   string
		decode func() (Page, error)
	}{
		{"v1", func() (Page, error) {
			return column.DecodeDataPageV1(DataPageHeaderV1{header: &format.DataPageHeader{
				NumValues: math.MaxInt32,
				Encoding:  format.ALP,
			}}, data, nil)
		}},
		{"v2", func() (Page, error) {
			return column.DecodeDataPageV2(DataPageHeaderV2{header: &format.DataPageHeaderV2{
				NumValues: math.MaxInt32,
				NumRows:   math.MaxInt32,
				Encoding:  format.ALP,
			}}, data, nil)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := test.decode(); !errors.Is(err, encoding.ErrInvalidArgument) {
				t.Fatalf("got %v, want wrapped encoding.ErrInvalidArgument", err)
			}
		})
	}
}

func TestALPPageHeaderPreflightErrorsRemainWrapped(t *testing.T) {
	column := &Column{typ: DoubleType}
	truncated := []byte{0}
	for _, test := range []struct {
		name   string
		decode func() (Page, error)
	}{
		{"v1", func() (Page, error) {
			return column.DecodeDataPageV1(DataPageHeaderV1{header: &format.DataPageHeader{NumValues: 1, Encoding: format.ALP}}, truncated, nil)
		}},
		{"v2", func() (Page, error) {
			return column.DecodeDataPageV2(DataPageHeaderV2{header: &format.DataPageHeaderV2{NumValues: 1, NumRows: 1, Encoding: format.ALP}}, truncated, nil)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := test.decode(); !errors.Is(err, encoding.ErrInvalidArgument) {
				t.Fatalf("got %v, want wrapped encoding.ErrInvalidArgument", err)
			}
		})
	}
}
