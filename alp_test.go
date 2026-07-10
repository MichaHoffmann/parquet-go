package parquet_test

import (
	"bytes"
	"io"
	"math"
	"testing"

	"github.com/parquet-go/parquet-go"
)

func TestEncodedALP(t *testing.T) {
	schema := parquet.NewSchema("row", parquet.Group{
		"v": parquet.Encoded(parquet.Leaf(parquet.DoubleType), &parquet.ALP),
	})
	src := make([]float64, 3000)
	for i := range src {
		src[i] = float64(i) * 0.5
	}
	rows := make([]parquet.Row, len(src))
	for i, value := range src {
		rows[i] = parquet.Row{parquet.ValueOf(value).Level(0, 0, 0)}
	}

	var buf bytes.Buffer
	w := parquet.NewGenericWriter[any](&buf, schema)
	if _, err := w.WriteRows(rows); err != nil {
		t.Fatalf("write rows: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	assertALPEncoding(t, buf.Bytes())

	f, err := parquet.OpenFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	got := readALPColumnBits(t, f.Root().Columns()[0])
	if len(got) != len(src) {
		t.Fatalf("got %d values, want %d", len(got), len(src))
	}
	for i := range src {
		if got[i] != math.Float64bits(src[i]) {
			t.Fatalf("row %d: got bits %#x want %#x", i, got[i], math.Float64bits(src[i]))
		}
	}
}

func readALPColumnBits(t *testing.T, column *parquet.Column) []uint64 {
	t.Helper()
	pages := column.Pages()
	defer pages.Close()
	var bits []uint64
	values := make([]parquet.Value, 1024)
	for {
		page, err := pages.ReadPage()
		if err == io.EOF {
			return bits
		}
		if err != nil {
			t.Fatalf("read page: %v", err)
		}
		reader := page.Values()
		for {
			n, err := reader.ReadValues(values)
			for _, value := range values[:n] {
				bits = append(bits, math.Float64bits(value.Double()))
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				parquet.Release(page)
				t.Fatalf("read values: %v", err)
			}
		}
		parquet.Release(page)
	}
}
