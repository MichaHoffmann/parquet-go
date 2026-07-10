package parquet_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/parquet-go/parquet-go"
)

type alpRow struct {
	D float64 `parquet:"d,alp"`
	F float32 `parquet:"f,alp"`
}

// genALPRows builds decimal-scaled values with occasional exceptions.
func genALPRows(n int) []alpRow {
	rows := make([]alpRow, n)
	for i := range rows {
		d := float64(i%100000)*0.01 - float64(i%7)*0.25
		f := float32(i%50000) * 0.1
		switch i % 9973 {
		case 0:
			d = math.NaN()
		case 1:
			d = math.Inf(1)
		case 2:
			f = float32(math.Inf(-1))
		case 3:
			d = math.Copysign(0, -1)
		case 4:
			f = float32(math.NaN())
		}
		rows[i] = alpRow{D: d, F: f}
	}
	return rows
}

// TestALPWriterRoundTrip verifies exact-bit round trips through the writer,
// including NaN and infinities.
func TestWriterALPRoundTrip(t *testing.T) {
	const n = 250_000 // many pages at the default 256 KiB page size
	rows := genALPRows(n)

	var buf bytes.Buffer
	w := parquet.NewGenericWriter[alpRow](&buf)
	if _, err := w.Write(rows); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	assertALPEncoding(t, buf.Bytes())

	r := parquet.NewGenericReader[alpRow](bytes.NewReader(buf.Bytes()))
	defer r.Close()
	got := make([]alpRow, n)
	read := 0
	for read < n {
		m, err := r.Read(got[read:])
		read += m
		if err != nil {
			if read == n {
				break
			}
			t.Fatalf("read after %d rows: %v", read, err)
		}
	}
	if read != n {
		t.Fatalf("read %d rows, want %d", read, n)
	}
	for i := range rows {
		if math.Float64bits(got[i].D) != math.Float64bits(rows[i].D) {
			t.Fatalf("D lossy at %d: got %v want %v", i, got[i].D, rows[i].D)
		}
		if math.Float32bits(got[i].F) != math.Float32bits(rows[i].F) {
			t.Fatalf("F lossy at %d: got %v want %v", i, got[i].F, rows[i].F)
		}
	}
}

func assertALPEncoding(t *testing.T, data []byte) {
	t.Helper()
	f, err := parquet.OpenFile(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("open for encoding check: %v", err)
	}
	for _, rowGroup := range f.Metadata().RowGroups {
		for _, column := range rowGroup.Columns {
			for _, enc := range column.MetaData.Encoding {
				if enc == parquet.ALP.Encoding() {
					return
				}
			}
		}
	}
	t.Fatal("written file does not declare ALP encoding")
}
